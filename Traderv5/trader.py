"""Live trading orchestration for Trader V5."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import logging
import threading

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

        try:
            self.executor.sync_trade_activity()
        except Exception as exc:  # pragma: no cover - logging only
            diagnostics_errors.append(f"sync_trade_activity:{exc}")

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
                    features=features,
                    price_snapshot=price_snapshot,
                    account=account_snapshot,
                    position=position_ctx,
                    open_positions=open_positions,
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
            positions=self._serialize_positions(open_positions_payload),
            day_trade=self._day_trade_status(),
            market_status=market_status,
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


__all__ = ["ModelDrivenTrader", "TraderV5Config"]
