"""Live trading orchestration for Trader V5."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import logging
import threading
import time

import pandas as pd

from Traderv4.funcs import (
    AccountSnapshot,
    EventLogger,
    RiskConfig,
    TradeDecision,
    TradeExecutor,
    TradeFrequencyTracker,
    YahooFinanceClient,
)
from Traderv4.state import (
    AlertRecord,
    CycleDiagnostics,
    DayTradeStatus,
    DecisionRecord,
    MarketStatus,
    PositionRecord,
    SentimentPoint,
    StateStore,
)

from Traderv5.data_sources import YahooFinanceDataFetcher, collect_gdelt_window
from Traderv5.decision import ModelDecisionEngine
from Traderv5.persistence import PositionPersistence
from Traderv5.features import FeatureEngineer, FeatureEngineerConfig
from Traderv5.model.predictor import ModelPredictor
from Traderv5.persistence import PersistentTradeJournal

_LOG = logging.getLogger("traderv5.trader")


@dataclass
class TraderV5Config:
    tickers: List[str]
    lookback_days: int = 7
    price_interval: str = "1h"
    gdelt_timeline_minutes: int = 60
    gdelt_delay: float = 0.75
    sentiment_window_minutes: int = 60
    cycle_pause_seconds: int = 300
    risk: RiskConfig = field(default_factory=RiskConfig)
    label_horizon_minutes: int = 1440
    positive_threshold: float = 0.0035
    negative_threshold: float = -0.0035

    def __post_init__(self) -> None:
        normalized: List[str] = []
        for ticker in self.tickers:
            value = str(ticker).strip().upper()
            if value and value not in normalized:
                normalized.append(value)
        self.tickers = normalized


@dataclass
class TradingContext:
    account: AccountSnapshot
    positions_by_ticker: Dict[str, Dict[str, Any]]
    open_positions: int
    market_open: bool
    market_status: Optional[MarketStatus]
    fetched_at: datetime

    def position_for(self, ticker: str) -> Optional[Dict[str, Any]]:
        return self.positions_by_ticker.get(ticker.upper())

    def register_entry(self, ticker: str, decision: TradeDecision) -> None:
        symbol = ticker.upper()
        action = decision.action.upper()
        side = "LONG" if action == "BUY" else "SHORT"
        try:
            quantity = abs(float(decision.quantity or 0.0))
        except (TypeError, ValueError):
            quantity = 0.0
        market_value = float(decision.notional)
        cost_basis = market_value if side == "LONG" else -market_value
        if symbol not in self.positions_by_ticker:
            self.open_positions += 1
        self.positions_by_ticker[symbol] = {
            "ticker": symbol,
            "side": side,
            "quantity": quantity,
            "market_value": market_value,
            "cost_basis": cost_basis,
        }

    def register_exit(self, ticker: str) -> None:
        symbol = ticker.upper()
        if symbol in self.positions_by_ticker:
            self.positions_by_ticker.pop(symbol, None)
            self.open_positions = max(0, self.open_positions - 1)

    def has_position(self, ticker: str) -> bool:
        position = self.position_for(ticker)
        if not position:
            return False
        try:
            quantity = abs(float(position.get("quantity", 0.0)))
        except (TypeError, ValueError):
            quantity = 0.0
        return quantity > 0


@dataclass
class TickerEvaluation:
    ticker: str
    decision: TradeDecision
    order_id: Optional[str]
    evaluated_at: datetime
    last_bar_at: Optional[datetime]
    price: Optional[float]
    position_side: Optional[str]
    position_quantity: float
    profit_pct: Optional[float]
    errors: List[str] = field(default_factory=list)


@dataclass
class TickerWatchState:
    ticker: str
    next_check: datetime
    last_bar_at: Optional[datetime] = None
    last_price: Optional[float] = None
    last_result: Optional[TickerEvaluation] = None

    def schedule_next(
        self,
        *,
        now: datetime,
        has_position: bool,
        flat_interval: timedelta,
        position_interval: timedelta,
        min_interval: timedelta,
    ) -> None:
        interval = position_interval if has_position else flat_interval
        if interval < min_interval:
            interval = min_interval
        self.next_check = now + interval


class ModelDrivenTrader:
    """Automation loop that fuses GDELT, Yahoo Finance, and ML scoring."""

    def __init__(
        self,
        config: TraderV5Config,
        alpaca_client,
        balance_fetcher,
        *,
        predictor: Optional[ModelPredictor] = None,
        state_store: Optional[StateStore] = None,
        feature_config: Optional[FeatureEngineerConfig] = None,
        trade_memory: Optional[PersistentTradeJournal] = None,
    ) -> None:
        self.config = config
        self.yahoo_intraday = YahooFinanceClient()
        self.yahoo_history = YahooFinanceDataFetcher()
        if feature_config is None:
            feature_config = FeatureEngineerConfig(
                label_horizon_minutes=config.label_horizon_minutes,
                positive_threshold=config.positive_threshold,
                negative_threshold=config.negative_threshold,
            )
        self.feature_engineer = FeatureEngineer(feature_config)
        self.predictor = predictor or ModelPredictor.load_default()
        self.trade_tracker = TradeFrequencyTracker(max_day_trades=config.risk.max_positions * 2)
        self.executor = TradeExecutor(alpaca_client, self.trade_tracker)
        self.logger = EventLogger()
        self.state_store = state_store or StateStore()
        self.trade_memory = trade_memory
        self.balance_fetcher = balance_fetcher
        self.decision_engine = ModelDecisionEngine(
            self.predictor,
            config.risk,
            self.trade_tracker,
            trade_memory=self.trade_memory,
        )
        self._lock = threading.Lock()
        self.sentiment_history: List[SentimentPoint] = []

    def run_cycle(self, tickers: Optional[List[str]] = None) -> None:
        with self._lock:
            self._run_cycle_impl(tickers)

    def _run_cycle_impl(self, tickers: Optional[List[str]] = None) -> None:
        cycle_started = datetime.utcnow()
        diagnostics_errors: List[str] = []
        alerts: List[AlertRecord] = []
        decisions: List[DecisionRecord] = []
        entries = exits = holds = 0
        tickers_processed = 0
        tickers_to_process = tickers or list(self.config.tickers)
        if not tickers_to_process:
            return

    def _prepare_context_impl(self) -> TradingContext:
        diagnostics_errors: List[str] = []
        try:
            self.executor.sync_trade_activity()
        except Exception as exc:  # pragma: no cover - logging only
            diagnostics_errors.append(f"sync_trade_activity:{exc}")
            _LOG.debug("sync_trade_activity failed: %s", exc)

        account_snapshot = self.executor.get_account_snapshot()
        if account_snapshot.cash == 0.0 and self.balance_fetcher is not None:
            try:
                fallback_cash = float(self.balance_fetcher())
            except Exception:  # pragma: no cover - best effort fallback
                fallback_cash = account_snapshot.cash
            else:
                account_snapshot = AccountSnapshot(
                    equity=account_snapshot.equity or fallback_cash,
                    cash=fallback_cash,
                    buying_power=account_snapshot.buying_power or fallback_cash,
                    portfolio_value=account_snapshot.portfolio_value or fallback_cash,
                )

        self.logger.log_trade_frequency(self.trade_tracker)

        open_positions_payload = self.executor.get_open_positions()
        if self.position_store:
            self.position_store.reconcile(open_positions_payload)
            self.decision_engine.sync_memory_from_store()
        positions_by_ticker: Dict[str, Dict[str, Any]] = {}
        for pos in open_positions_payload:
            key = str(
                pos.get("ticker")
                or pos.get("symbol")
                or pos.get("asset_id")
                or pos.get("id", "")
            ).upper()
            if not key:
                continue
            payload = dict(pos)
            if "ticker" not in payload:
                payload["ticker"] = key
            positions_by_ticker[key] = payload
        open_positions = len(positions_by_ticker)
        if self.trade_memory is not None:
            self.trade_memory.sync_broker_positions(positions_by_ticker)

        market_clock = self.executor.get_market_clock()
        market_open = True
        market_status: Optional[MarketStatus] = None
        if market_clock:
            market_open = bool(market_clock.get("is_open", True))
            market_status = MarketStatus(
                is_open=market_open,
                next_open=market_clock.get("next_open"),
                next_close=market_clock.get("next_close"),
                timestamp=market_clock.get("timestamp", datetime.utcnow()),
            )
            if not market_open:
                self.logger.log_market_closed(market_clock.get("next_open"))

        context = TradingContext(
            account=account_snapshot,
            positions_by_ticker=positions_by_ticker,
            open_positions=open_positions,
            market_open=market_open,
            market_status=market_status,
            fetched_at=datetime.utcnow(),
        )
        if diagnostics_errors:
            _LOG.debug("Context diagnostics: %s", diagnostics_errors)
        return context

    def _process_ticker_impl(
        self,
        ticker: str,
        context: TradingContext,
        *,
        start: datetime,
        end: datetime,
    ) -> TickerEvaluation:
        ticker_symbol = ticker.upper()
        evaluation_started = datetime.utcnow()
        errors: List[str] = []

        try:
            price_df = self._fetch_price_window(ticker_symbol, start, end)
        except Exception as exc:
            errors.append(f"price_fetch:{ticker_symbol}:{exc}")
            decision = self._make_hold_decision(ticker_symbol, "price fetch failed")
            return TickerEvaluation(
                ticker=ticker_symbol,
                decision=decision,
                order_id=None,
                evaluated_at=evaluation_started,
                last_bar_at=None,
                price=None,
                position_side=None,
                position_quantity=0.0,
                profit_pct=None,
                errors=errors,
            )

        if price_df.empty:
            errors.append(f"no_price:{ticker_symbol}")
            decision = self._make_hold_decision(ticker_symbol, "no price data")
            return TickerEvaluation(
                ticker=ticker_symbol,
                decision=decision,
                order_id=None,
                evaluated_at=evaluation_started,
                last_bar_at=None,
                price=None,
                position_side=None,
                position_quantity=0.0,
                profit_pct=None,
                errors=errors,
            )

        last_index = price_df.index[-1]
        if hasattr(last_index, "to_pydatetime"):
            last_bar_at = last_index.to_pydatetime()
        else:
            last_bar_at = pd.Timestamp(last_index).to_pydatetime()
        try:
            last_bar_at = last_bar_at.astimezone(timezone.utc)
        except Exception:
            last_bar_at = last_bar_at.replace(tzinfo=timezone.utc)

        try:
            gdelt_window = collect_gdelt_window(
                ticker_symbol,
                start=start.replace(tzinfo=None),
                end=end.replace(tzinfo=None),
                timeline_minutes=self.config.gdelt_timeline_minutes,
                delay=self.config.gdelt_delay,
            )
        except Exception as exc:
            errors.append(f"gdelt:{ticker_symbol}:{exc}")
            decision = self._make_hold_decision(ticker_symbol, "gdelt fetch failed")
            return TickerEvaluation(
                ticker=ticker_symbol,
                decision=decision,
                order_id=None,
                evaluated_at=evaluation_started,
                last_bar_at=last_bar_at,
                price=float(price_df["close"].iloc[-1]),
                position_side=None,
                position_quantity=0.0,
                profit_pct=None,
                errors=errors,
            )

        feature_frame = self.feature_engineer.build_training_frame(
            ticker_symbol,
            price_df,
            gdelt_window,
            include_labels=False,
        )
        if feature_frame.empty:
            errors.append(f"no_features:{ticker_symbol}")
            decision = self._make_hold_decision(ticker_symbol, "no features available")
            return TickerEvaluation(
                ticker=ticker_symbol,
                decision=decision,
                order_id=None,
                evaluated_at=evaluation_started,
                last_bar_at=last_bar_at,
                price=float(price_df["close"].iloc[-1]),
                position_side=None,
                position_quantity=0.0,
                profit_pct=None,
                errors=errors,
            )

        latest_row = feature_frame.iloc[-1]
        feature_columns = self.feature_engineer.feature_columns(feature_frame)
        features = latest_row[feature_columns].to_dict()
        price_snapshot = {
            "close": float(latest_row.get("close", price_df["close"].iloc[-1])),
            "volume": float(
                latest_row.get(
                    "volume",
                    price_df["volume"].iloc[-1] if "volume" in price_df else 0.0,
                )
            ),
        }

        position_ctx = context.position_for(ticker_symbol)
        decision = self.decision_engine.decide(
            ticker_symbol,
            features=features,
            price_snapshot=price_snapshot,
            account=context.account,
            position=position_ctx,
            open_positions=context.open_positions,
        )
        self.logger.log_decision(decision)

        probability = self._extract_probability(decision.metadata)
        article_count = int(latest_row.get("article_count_60", 0))
        self.sentiment_history.append(
            SentimentPoint(
                seen_at=datetime.utcnow(),
                average_sentiment=probability,
                sentiment_delta=probability - 0.5,
                article_count=article_count,
                ticker=ticker_symbol,
            )
        )

        order_id: Optional[str] = None
        if decision.action != "HOLD":
            order_id = self.executor.place_trade(decision)
            if order_id:
                if decision.intent == "entry":
                    context.register_entry(ticker_symbol, decision)
                elif decision.intent == "exit":
                    context.register_exit(ticker_symbol)

        position_after = context.position_for(ticker_symbol)
        position_side: Optional[str]
        position_quantity: float
        if position_after:
            position_side = str(position_after.get("side", "")).upper()
            try:
                position_quantity = abs(float(position_after.get("quantity", 0.0)))
            except (TypeError, ValueError):
                position_quantity = 0.0
        else:
            position_side = None
            position_quantity = 0.0

        profit_pct = self._position_profit_pct(position_after, price_snapshot.get("close"))

        return TickerEvaluation(
            ticker=ticker_symbol,
            decision=decision,
            order_id=order_id,
            evaluated_at=evaluation_started,
            last_bar_at=last_bar_at,
            price=price_snapshot.get("close"),
            position_side=position_side,
            position_quantity=position_quantity,
            profit_pct=profit_pct,
            errors=errors,
        )

    @staticmethod
    def _make_hold_decision(
        ticker: str, reason: str, metadata: Optional[Dict[str, str]] = None
    ) -> TradeDecision:
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

    @staticmethod
    def _extract_probability(metadata: Optional[Dict[str, Any]]) -> float:
        if not metadata:
            return 0.5
        candidates = [
            metadata.get("probability_long"),
            metadata.get("prob_long"),
            metadata.get("signal_probability"),
        ]
        for value in candidates:
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return 0.5

    @staticmethod
    def _position_profit_pct(
        position: Optional[Dict[str, Any]], price: Optional[float]
    ) -> Optional[float]:
        if not position or price is None:
            return None
        try:
            quantity = abs(float(position.get("quantity", 0.0)))
        except (TypeError, ValueError):
            return None
        if quantity <= 0:
            return None
        cost_basis_raw = position.get("cost_basis")
        try:
            cost_basis = float(cost_basis_raw) if cost_basis_raw is not None else None
        except (TypeError, ValueError):
            cost_basis = None
        if cost_basis in (None, 0.0):
            return None
        entry_price = abs(cost_basis) / quantity if quantity > 0 else None
        if not entry_price or entry_price <= 0:
            return None
        side = str(position.get("side", "")).upper()
        if side in {"SHORT", "SELL"}:
            return (entry_price - price) / entry_price
        return (price - entry_price) / entry_price

    def run_cycle(self) -> None:
        with self._lock:
            self._run_cycle_impl()

    def _run_cycle_impl(self) -> None:
        cycle_started = datetime.utcnow()
        diagnostics_errors: List[str] = []
        alerts: List[AlertRecord] = []
        decisions: List[DecisionRecord] = []
        entries = exits = holds = 0
        tickers_processed = 0

        context = self.prepare_context(locked=True)

        end = datetime.utcnow().replace(tzinfo=timezone.utc)
        start = end - timedelta(days=self.config.lookback_days)

        for ticker in tickers_to_process:
            if not market_open:
                continue
            tickers_processed += 1
            marked = False
            try:
                price_df = self._fetch_price_window(ticker, start, end)
                if price_df.empty:
                    diagnostics_errors.append(f"no_price:{ticker}")
                    if self.trade_memory is not None and not marked:
                        self.trade_memory.mark_evaluated(ticker)
                        marked = True
                    continue
                gdelt_window = collect_gdelt_window(
                    ticker,
                    start=start.replace(tzinfo=None),
                    end=end.replace(tzinfo=None),
                    timeline_minutes=self.config.gdelt_timeline_minutes,
                    delay=self.config.gdelt_delay,
                )
                feature_frame = self.feature_engineer.build_training_frame(
                    ticker,
                    price_df,
                    gdelt_window,
                    include_labels=False,
                )
                if feature_frame.empty:
                    diagnostics_errors.append(f"no_features:{ticker}")
                    if self.trade_memory is not None and not marked:
                        self.trade_memory.mark_evaluated(ticker)
                        marked = True
                    continue
                latest_row = feature_frame.iloc[-1]
                feature_columns = self.feature_engineer.feature_columns(feature_frame)
                features = latest_row[feature_columns].to_dict()
                price_snapshot = {
                    "close": float(latest_row.get("close", price_df["close"].iloc[-1])),
                    "volume": float(latest_row.get("volume", price_df["volume"].iloc[-1] if "volume" in price_df else 0.0)),
                }
                if self.trade_memory is not None:
                    self.trade_memory.update_market_price(ticker, price_snapshot.get("close", 0.0) or 0.0)
                position_ctx = positions_by_ticker.get(ticker)
                decision = self.decision_engine.decide(
                    ticker,
                    context,
                    start=start,
                    end=end,
                    locked=True,
                )
                self.logger.log_decision(decision)
                decisions.append(DecisionRecord.from_decision(decision))
                metadata = decision.metadata or {}
                probability = float(metadata.get("probability_long", features.get("probability_long", 0.5)))
                self.sentiment_history.append(
                    SentimentPoint(
                        seen_at=datetime.utcnow(),
                        average_sentiment=probability,
                        sentiment_delta=probability - 0.5,
                        article_count=int(latest_row.get("article_count_60", 0)),
                        ticker=ticker,
                    )
                )
                if decision.action == "HOLD":
                    holds += 1
                    if self.trade_memory is not None:
                        self.trade_memory.touch(ticker)
                        self.trade_memory.mark_evaluated(ticker)
                        marked = True
                    continue
                order_id = self.executor.place_trade(decision)
                if order_id:
                    if decision.intent == "entry":
                        entries += 1
                        open_positions += 1
                    elif decision.intent == "exit":
                        exits += 1
                        open_positions = max(0, open_positions - 1)
                if self.trade_memory is not None:
                    self.trade_memory.mark_evaluated(ticker)
                    marked = True
            except Exception as exc:
                diagnostics_errors.append(f"{ticker}:{exc}")
                _LOG.exception("Cycle error for %s", ticker)
                continue
            finally:
                if self.trade_memory is not None and not marked:
                    self.trade_memory.mark_evaluated(ticker)

            decisions.append(DecisionRecord.from_decision(result.decision))
            if result.decision.action == "HOLD":
                holds += 1
            elif result.decision.intent == "entry":
                entries += 1
            elif result.decision.intent == "exit":
                exits += 1
            diagnostics_errors.extend(result.errors)

        cycle_completed = datetime.utcnow()
        diagnostics = CycleDiagnostics(
            started_at=cycle_started,
            completed_at=cycle_completed,
            tickers_processed=tickers_processed,
            alerts_generated=len(alerts),
            decisions_made=len(decisions),
            holds=holds,
            entries=entries,
            exits=exits,
            errors=diagnostics_errors,
        )
        self.state_store.update(
            alerts=alerts,
            decisions=decisions,
            sentiment_history=self.sentiment_history[-300:],
            positions=self._serialize_positions(list(context.positions_by_ticker.values())),
            day_trade=self._day_trade_status(),
            market_status=context.market_status,
            diagnostics=diagnostics,
        )
        if self.trade_memory is not None:
            self.trade_memory.flush()

    def _fetch_price_window(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        lookback = end - start
        interval = self.config.price_interval
        if interval.endswith("m"):
            interval_str = interval
        elif interval.endswith("h"):
            interval_str = interval
        else:
            interval_str = "1h"
        frame = self.yahoo_history.fetch_price_history(
            ticker,
            range_=self._range_from_lookback(lookback),
            interval=interval_str,
        )
        if frame.empty:
            return frame
        frame = frame[(frame.index >= start.astimezone(timezone.utc)) & (frame.index <= end.astimezone(timezone.utc))]
        return frame

    @staticmethod
    def _range_from_lookback(lookback: timedelta) -> str:
        days = max(1, int(lookback.total_seconds() // 86400))
        if days <= 7:
            return "7d"
        if days <= 30:
            return "1mo"
        if days <= 90:
            return "3mo"
        if days <= 180:
            return "6mo"
        if days <= 365:
            return "1y"
        return "max"

    def _serialize_positions(self, payload: List[Dict[str, Any]]) -> List[PositionRecord]:
        records: List[PositionRecord] = []
        for item in payload:
            try:
                quantity = float(item.get("qty", item.get("quantity", 0.0)))
            except (TypeError, ValueError):
                quantity = 0.0
            try:
                market_value = float(item.get("market_value", 0.0))
            except (TypeError, ValueError):
                market_value = 0.0
            record = PositionRecord(
                ticker=item.get("ticker", "").upper(),
                side=item.get("side", "").upper(),
                quantity=quantity,
                market_value=market_value,
                cost_basis=None,
                last_updated=datetime.utcnow(),
            )
            records.append(record)
        return records

    def _day_trade_status(self):
        used = self.trade_tracker.intraday_trades()
        next_reset = self.trade_tracker.next_reset_date()
        return DayTradeStatus(used=used, limit=self.trade_tracker.max_day_trades, next_reset=next_reset)


class ReactiveTraderRunner:
    def __init__(
        self,
        trader: ModelDrivenTrader,
        *,
        status_interval: int = 90,
        sync_interval: int = 180,
        flat_interval_seconds: Optional[int] = None,
        position_interval_seconds: Optional[int] = None,
    ) -> None:
        self.trader = trader
        self.logger = logging.getLogger("traderv5.reactive")
        self.status_interval = max(30, int(status_interval))
        self.sync_interval = max(60, int(sync_interval))
        base_interval = self._resolve_interval_seconds(trader.config.price_interval)
        flat_seconds = (
            flat_interval_seconds
            if flat_interval_seconds is not None
            else max(base_interval // 2, 180)
        )
        position_seconds = (
            position_interval_seconds
            if position_interval_seconds is not None
            else max(base_interval // 4, 90)
        )
        self.flat_interval = timedelta(seconds=flat_seconds)
        self.position_interval = timedelta(seconds=position_seconds)
        self.min_interval = timedelta(seconds=45)
        self._stop = threading.Event()
        now = datetime.utcnow()
        self._watchers: Dict[str, TickerWatchState] = {
            ticker: TickerWatchState(ticker=ticker, next_check=now)
            for ticker in trader.config.tickers
        }

    def stop(self) -> None:
        self._stop.set()

    def run_forever(self) -> None:
        status_deadline = datetime.utcnow() + timedelta(seconds=self.status_interval)
        next_sync = datetime.utcnow()
        while not self._stop.is_set():
            now = datetime.utcnow()
            if now >= next_sync:
                try:
                    self.trader.executor.sync_trade_activity()
                except Exception as exc:  # pragma: no cover - diagnostics only
                    self.logger.debug("activity sync failed: %s", exc)
                next_sync = now + timedelta(seconds=self.sync_interval)

            due_tickers = [
                ticker
                for ticker, state in self._watchers.items()
                if now >= state.next_check
            ]

            if not due_tickers:
                sleep_until = min(
                    (state.next_check for state in self._watchers.values()),
                    default=now + self.flat_interval,
                )
                wait_seconds = max(
                    1.0,
                    min(60.0, (sleep_until - now).total_seconds()),
                )
                self._stop.wait(timeout=wait_seconds)
                continue

            context = self.trader.prepare_context()
            if not context.market_open:
                next_open = (
                    context.market_status.next_open
                    if context.market_status and context.market_status.next_open
                    else None
                )
                resume = (
                    next_open - timedelta(minutes=5)
                    if next_open
                    else datetime.utcnow() + timedelta(minutes=15)
                )
                for state in self._watchers.values():
                    if resume > state.next_check:
                        state.next_check = resume
                self._stop.wait(timeout=60)
                continue

            end = datetime.utcnow().replace(tzinfo=timezone.utc)
            start = end - timedelta(days=self.trader.config.lookback_days)

            for ticker in due_tickers:
                if self._stop.is_set():
                    break
                result = self.trader.process_ticker(
                    ticker,
                    context,
                    start=start,
                    end=end,
                )
                state = self._watchers[ticker]
                state.last_bar_at = result.last_bar_at
                state.last_price = result.price
                state.last_result = result
                has_position = (
                    result.position_side in {"LONG", "SHORT"}
                    and result.position_quantity > 0
                )
                state.schedule_next(
                    now=datetime.utcnow(),
                    has_position=has_position,
                    flat_interval=self.flat_interval,
                    position_interval=self.position_interval,
                    min_interval=self.min_interval,
                )

            if self._stop.is_set():
                break

            if datetime.utcnow() >= status_deadline:
                summary = ", ".join(
                    f"{ticker}:{int(max(0, (state.next_check - datetime.utcnow()).total_seconds()))}s"
                    for ticker, state in self._watchers.items()
                )
                self.logger.info("Reactive loop active | next checks %s", summary)
                status_deadline = datetime.utcnow() + timedelta(seconds=self.status_interval)

        self.logger.info("Reactive loop stopped")

    @staticmethod
    def _resolve_interval_seconds(interval: str) -> int:
        if not interval:
            return 300
        text = interval.strip().lower()
        digits = "".join(ch for ch in text if ch.isdigit())
        try:
            value = int(digits) if digits else 1
        except ValueError:
            value = 1
        if text.endswith("m"):
            return max(60, value * 60)
        if text.endswith("h"):
            return max(3600, value * 3600)
        if text.endswith("d"):
            return max(86400, value * 86400)
        return max(60, value * 60)


__all__ = [
    "ModelDrivenTrader",
    "TraderV5Config",
    "ReactiveTraderRunner",
    "TradingContext",
    "TickerEvaluation",
]
