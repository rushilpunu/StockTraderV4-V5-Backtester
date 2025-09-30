from __future__ import annotations

import logging
import os
import signal
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Traderv4.GDELT import GDELTClient
from Traderv4.funcs import (
    AccountSnapshot,
    DecisionEngine,
    EventLogger,
    RiskConfig,
    SentimentProcessor,
    TradeExecutor,
    TradeFrequencyTracker,
    VolatilityAnalyzer,
    VolatilityThresholds,
    YahooFinanceClient,
)
from config.credentials import load_alpaca_credentials
from Traderv4.state import (
    AlertRecord,
    DayTradeStatus,
    DecisionRecord,
    CycleDiagnostics,
    MarketStatus,
    PositionRecord,
    SentimentPoint,
    StateStore,
)


@dataclass
class TraderConfig:
    tickers: List[str]
    gdelt_minutes_back: int = 180
    sentiment_window: timedelta = timedelta(hours=1)
    cycle_pause_seconds: int = 300
    thresholds: VolatilityThresholds = field(default_factory=VolatilityThresholds)
    risk: RiskConfig = field(default_factory=RiskConfig)

    def __post_init__(self) -> None:
        normalized = []
        seen: set[str] = set()
        for ticker in self.tickers:
            value = str(ticker).strip().upper()
            if value and value not in seen:
                normalized.append(value)
                seen.add(value)
        self.tickers = normalized


class NewsSentimentTrader:
    def __init__(
        self,
        config: TraderConfig,
        alpaca_client,
        balance_fetcher: Callable[[], float],
        state_store: Optional[StateStore] = None,
    ):
        self.config = config
        self.gdelt = GDELTClient()
        self.yahoo = YahooFinanceClient()
        self.processor = SentimentProcessor()
        self.analyzer = VolatilityAnalyzer(config.thresholds)
        self.trade_tracker = TradeFrequencyTracker()
        self.decision_engine = DecisionEngine(config.risk, self.trade_tracker)
        self.executor = TradeExecutor(alpaca_client, self.trade_tracker)
        self.logger = EventLogger()
        self.balance_fetcher = balance_fetcher
        self.baselines: Dict[str, Dict[str, float]] = {}
        self.state_store = state_store or StateStore()
        self.sentiment_history: Deque[SentimentPoint] = deque(maxlen=300)
        self._config_lock = threading.Lock()

    def run_cycle(self) -> None:
        cycle_started = datetime.utcnow()
        diagnostics_errors: List[str] = []
        self.executor.sync_trade_activity()
        account_snapshot = self.executor.get_account_snapshot()
        if account_snapshot.cash == 0.0 and self.balance_fetcher is not None:
            try:
                fallback_cash = float(self.balance_fetcher())
            except Exception:
                fallback_cash = account_snapshot.cash
            else:
                account_snapshot = AccountSnapshot(
                    equity=account_snapshot.equity or fallback_cash,
                    cash=fallback_cash,
                    buying_power=account_snapshot.buying_power or fallback_cash,
                    portfolio_value=account_snapshot.portfolio_value or fallback_cash,
                )
        self.logger.log_trade_frequency(self.trade_tracker)
        cycle_alerts: List[AlertRecord] = []
        cycle_decisions: List[DecisionRecord] = []
        tickers_processed = 0
        holds = 0
        entries = 0
        exits = 0
        initial_positions_payload = self.executor.get_open_positions()
        positions_by_ticker: Dict[str, Dict[str, Any]] = {}
        for item in initial_positions_payload:
            ticker_key = str(item.get("ticker", "")).upper()
            if not ticker_key:
                continue
            try:
                qty = float(item.get("quantity", 0.0))
            except (TypeError, ValueError):
                qty = 0.0
            if abs(qty) < 1e-6:
                continue
            positions_by_ticker[ticker_key] = item
        open_positions = len(positions_by_ticker)
        market_clock_raw = self.executor.get_market_clock()
        market_open = True
        next_open_time: Optional[datetime] = None
        market_status: Optional[MarketStatus]
        if market_clock_raw:
            market_open = bool(market_clock_raw.get("is_open", True))
            next_open_time = market_clock_raw.get("next_open")
            market_status = MarketStatus(
                is_open=market_open,
                next_open=market_clock_raw.get("next_open"),
                next_close=market_clock_raw.get("next_close"),
                timestamp=market_clock_raw.get("timestamp", datetime.utcnow()),
            )
            if not market_open:
                self.logger.log_market_closed(next_open_time)
        else:
            market_status = None
        for ticker in self.config.tickers:
            if not market_open:
                continue
            tickers_processed += 1
            ticker_upper = ticker.upper()
            try:
                baseline = self.baselines.get(ticker_upper) or self.baselines.get(ticker)
                if not baseline:
                    baseline = self.yahoo.fetch_volatility_baseline(ticker_upper)
                    self.baselines[ticker_upper] = baseline
                articles = self.gdelt.fetch_recent_articles(ticker_upper, minutes_back=self.config.gdelt_minutes_back)
                normalized = self.processor.from_gdelt_articles(articles)
                if not normalized:
                    continue
                grouped = self.processor.group_by_ticker(normalized)
                snapshots = self.processor.aggregate_sentiment(grouped, self.config.sentiment_window)
                for snapshot in snapshots:
                    alert = self.analyzer.evaluate(snapshot, baseline=baseline)
                    if not alert:
                        continue
                    self.logger.log_alert(alert)
                    cycle_alerts.append(AlertRecord.from_alert(alert))
                    self.sentiment_history.append(
                        SentimentPoint(
                            seen_at=alert.created_at,
                            average_sentiment=alert.average_sentiment,
                            sentiment_delta=alert.sentiment_delta,
                            article_count=alert.article_count,
                            ticker=alert.ticker,
                        )
                    )
                    price_snapshot = self.yahoo.fetch_intraday_snapshot(alert.ticker)
                    price_value_raw = price_snapshot.get("close") if price_snapshot else None
                    try:
                        last_price = float(price_value_raw) if price_value_raw is not None else 0.0
                    except (TypeError, ValueError):
                        last_price = 0.0
                    position_context = positions_by_ticker.get(alert.ticker.upper())
                    decision = self.decision_engine.decide(
                        alert=alert,
                        account=account_snapshot,
                        price_snapshot=price_snapshot,
                        position=position_context,
                        open_positions=open_positions,
                    )
                    self.logger.log_decision(decision)
                    cycle_decisions.append(DecisionRecord.from_decision(decision))
                    if decision.action == "HOLD":
                        holds += 1
                        continue
                    self.executor.place_trade(decision)
                    if decision.intent == "entry":
                        entries += 1
                        open_positions += 1
                        implied_qty = decision.quantity
                        if implied_qty is None and last_price > 0:
                            implied_qty = decision.notional / last_price
                        positions_by_ticker[alert.ticker.upper()] = {
                            "ticker": alert.ticker.upper(),
                            "side": "LONG" if decision.action == "BUY" else "SHORT",
                            "quantity": implied_qty or 0.0,
                            "market_value": decision.notional,
                        }
                    elif decision.intent == "exit" and open_positions > 0:
                        exits += 1
                        open_positions -= 1
                        positions_by_ticker.pop(alert.ticker.upper(), None)
            except Exception as exc:  # pragma: no cover - runtime diagnostics
                diagnostics_errors.append(f"{ticker_upper}: {exc}")
                logging.getLogger("trader").exception("Ticker %s failed", ticker)
                continue
        self.executor.sync_trade_activity()
        positions_payload = self.executor.get_open_positions()
        positions: List[PositionRecord] = []
        timestamp = datetime.utcnow()
        for item in positions_payload:
            positions.append(
                PositionRecord(
                    ticker=item.get("ticker", ""),
                    side=item.get("side", "").upper(),
                    quantity=float(item.get("quantity", 0.0)),
                    market_value=float(item.get("market_value", 0.0)),
                    cost_basis=(float(item["cost_basis"]) if item.get("cost_basis") is not None else None),
                    last_updated=timestamp,
                )
            )

        next_reset = self.trade_tracker.next_reset_date()
        status = DayTradeStatus(
            used=self.trade_tracker.intraday_trades(),
            limit=self.trade_tracker.max_day_trades,
            next_reset=next_reset,
        )
        diagnostics = CycleDiagnostics(
            started_at=cycle_started,
            completed_at=datetime.utcnow(),
            tickers_processed=tickers_processed,
            alerts_generated=len(cycle_alerts),
            decisions_made=len(cycle_decisions),
            holds=holds,
            entries=entries,
            exits=exits,
            errors=diagnostics_errors,
        )
        self.logger.log_cycle_summary(
            tickers=diagnostics.tickers_processed,
            alerts=diagnostics.alerts_generated,
            decisions=diagnostics.decisions_made,
            entries=diagnostics.entries,
            exits=diagnostics.exits,
            holds=diagnostics.holds,
            errors=len(diagnostics.errors),
        )
        self.state_store.update(
            alerts=cycle_alerts,
            decisions=cycle_decisions,
            positions=positions,
            day_trade=status,
            sentiment_history=list(self.sentiment_history),
            market_status=market_status,
            diagnostics=diagnostics,
        )

    def run_forever(self, stop_event: Optional[threading.Event] = None) -> None:
        while True:
            if stop_event and stop_event.is_set():
                break
            try:
                self.run_cycle()
            except Exception as exc:
                logging.getLogger("trader").exception("Cycle failed: %s", exc)
            if stop_event:
                if stop_event.wait(self.config.cycle_pause_seconds):
                    break
            else:
                time.sleep(self.config.cycle_pause_seconds)

    def get_settings(self) -> dict:
        with self._config_lock:
            return {
                "tickers": list(self.config.tickers),
                "gdeltMinutesBack": self.config.gdelt_minutes_back,
                "sentimentWindowMinutes": int(self.config.sentiment_window.total_seconds() // 60),
                "cyclePauseSeconds": self.config.cycle_pause_seconds,
                "thresholds": {
                    "minArticles": self.config.thresholds.min_articles,
                    "sentimentSpike": self.config.thresholds.sentiment_spike,
                    "sentimentExtreme": self.config.thresholds.sentiment_extreme,
                    "averageSentimentFloor": self.config.thresholds.average_sentiment_floor,
                    "singleArticleSpike": self.config.thresholds.single_article_spike,
                },
                "risk": {
                    "maxCapitalFraction": self.config.risk.max_capital_fraction,
                    "stopLossPct": self.config.risk.stop_loss_pct,
                    "takeProfitPct": self.config.risk.take_profit_pct,
                    "cooldownMinutes": self.config.risk.cooldown_minutes,
                    "entrySentimentThreshold": self.config.risk.entry_sentiment_threshold,
                    "exitSentimentThreshold": self.config.risk.exit_sentiment_threshold,
                    "maxPositionValue": self.config.risk.max_position_value,
                    "allowShorting": self.config.risk.allow_shorting,
                },
            }

    def update_settings(self, payload: Dict[str, Any]) -> dict:
        with self._config_lock:
            if "tickers" in payload:
                tickers = payload["tickers"]
                if isinstance(tickers, list) and tickers:
                    self.config.tickers = [str(t).upper() for t in tickers]
            if "gdeltMinutesBack" in payload:
                self.config.gdelt_minutes_back = int(payload["gdeltMinutesBack"])
            if "sentimentWindowMinutes" in payload:
                minutes = int(payload["sentimentWindowMinutes"])
                self.config.sentiment_window = timedelta(minutes=minutes)
            if "cyclePauseSeconds" in payload:
                self.config.cycle_pause_seconds = int(payload["cyclePauseSeconds"])

            thresholds = payload.get("thresholds")
            if isinstance(thresholds, dict):
                self.config.thresholds.min_articles = int(thresholds.get("minArticles", self.config.thresholds.min_articles))
                self.config.thresholds.sentiment_spike = float(thresholds.get("sentimentSpike", self.config.thresholds.sentiment_spike))
                self.config.thresholds.sentiment_extreme = float(thresholds.get("sentimentExtreme", self.config.thresholds.sentiment_extreme))
                if "averageSentimentFloor" in thresholds:
                    self.config.thresholds.average_sentiment_floor = float(thresholds.get("averageSentimentFloor", self.config.thresholds.average_sentiment_floor))
                if "singleArticleSpike" in thresholds:
                    self.config.thresholds.single_article_spike = float(thresholds.get("singleArticleSpike", self.config.thresholds.single_article_spike))

            risk = payload.get("risk")
            if isinstance(risk, dict):
                self.config.risk.max_capital_fraction = float(risk.get("maxCapitalFraction", self.config.risk.max_capital_fraction))
                self.config.risk.stop_loss_pct = float(risk.get("stopLossPct", self.config.risk.stop_loss_pct))
                self.config.risk.take_profit_pct = float(risk.get("takeProfitPct", self.config.risk.take_profit_pct))
                self.config.risk.cooldown_minutes = int(risk.get("cooldownMinutes", self.config.risk.cooldown_minutes))
                if "entrySentimentThreshold" in risk:
                    self.config.risk.entry_sentiment_threshold = float(risk.get("entrySentimentThreshold", self.config.risk.entry_sentiment_threshold))
                if "exitSentimentThreshold" in risk:
                    self.config.risk.exit_sentiment_threshold = float(risk.get("exitSentimentThreshold", self.config.risk.exit_sentiment_threshold))
                if "maxPositionValue" in risk:
                    raw_value = risk.get("maxPositionValue")
                    self.config.risk.max_position_value = None if raw_value in (None, "") else float(raw_value)
                if "allowShorting" in risk:
                    self.config.risk.allow_shorting = bool(risk.get("allowShorting"))

        return self.get_settings()


def _load_alpaca_client():
    try:
        from alpaca_trade_api import REST  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("alpaca-trade-api library is required") from exc
    creds = load_alpaca_credentials()
    return REST(key_id=creds.api_key, secret_key=creds.api_secret, base_url=creds.base_url)


def _load_account_balance(alpaca_client) -> float:
    account = alpaca_client.get_account()
    return float(account.cash)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    alpaca = _load_alpaca_client()
    state_store = StateStore()
    trader = NewsSentimentTrader(
        config=TraderConfig(tickers=default_watchlist()),
        alpaca_client=alpaca,
        balance_fetcher=lambda: _load_account_balance(alpaca),
        state_store=state_store,
    )

    stop_event = threading.Event()

    def _handle_signal(signum, _frame) -> None:
        logging.getLogger("trader").info("Received signal %s, shutting down...", signum)
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handle_signal)
        except (ValueError, AttributeError):
            continue

    logging.getLogger("trader").info("Starting trader loop (press Ctrl+C to stop)")
    try:
        trader.run_forever(stop_event=stop_event)
    except KeyboardInterrupt:
        stop_event.set()
    logging.getLogger("trader").info("Trader loop exited")

def _parse_watchlist(raw: str) -> List[str]:
    tickers = [ticker.strip().upper() for ticker in raw.split(",") if ticker.strip()]
    return [ticker for ticker in tickers if ticker]


def default_watchlist() -> List[str]:
    env_value = os.getenv("TRADERV4_TICKERS")
    if env_value:
        parsed = _parse_watchlist(env_value)
        if parsed:
            return parsed
    return [
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "TSLA",
        "NVDA",
        "META",
        "NFLX",
        "AMD",
        "BA",
        "CRM",
        "JPM",
        "DIS",
        "INTC",
        "PYPL",
    ]
