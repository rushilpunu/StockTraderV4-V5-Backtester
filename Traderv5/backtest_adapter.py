"""Backtester adapter for the Trader V5 model-driven strategy."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd

from Traderv4.funcs import AccountSnapshot, RiskConfig, TradeDecision, TradeFrequencyTracker

from backtester.bots.base import BaseBot, register_bot
from backtester.config import BacktestConfig
from backtester.data_sources import fetch_price_bars

from Traderv5.data_sources import collect_gdelt_window
from Traderv5.decision import ModelDecisionEngine
from Traderv5.features import FeatureEngineer
from Traderv5.model.predictor import ModelPredictor


@register_bot
class TraderV5Bot(BaseBot):
    name = "traderv5"

    def __init__(self) -> None:
        self.feature_engineer = FeatureEngineer()
        try:
            self.predictor = ModelPredictor.load_default()
        except FileNotFoundError as exc:
            raise RuntimeError(
                "TraderV5 model artifact not found. Run Traderv5/model/training.py to train and persist the model before backtesting."
            ) from exc
        self.risk = RiskConfig()
        self.trade_tracker = TradeFrequencyTracker(max_day_trades=10_000)
        self.engine = ModelDecisionEngine(self.predictor, self.risk, self.trade_tracker)
        self._feature_frame: Optional[pd.DataFrame] = None
        self._feature_columns = self.feature_engineer.feature_columns()
        self._volume_series: Optional[pd.Series] = None
        self._ticker: Optional[str] = None
        self._config: Optional[BacktestConfig] = None

    def setup(self, config: BacktestConfig, baseline: Dict[str, float]) -> None:
        self._config = config
        self.risk.max_capital_fraction = min(0.35, max(0.1, config.starting_cash / 1_000_000))
        self.risk.entry_sentiment_threshold = max(0.08, self.risk.entry_sentiment_threshold)
        self.risk.exit_sentiment_threshold = max(0.05, self.risk.exit_sentiment_threshold)
        self.trade_tracker = TradeFrequencyTracker(max_day_trades=10_000)
        self.engine = ModelDecisionEngine(self.predictor, self.risk, self.trade_tracker)
        self._feature_frame = None
        self._ticker = None

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
        if self._ticker != ticker:
            self._prepare_features(ticker)
        if self._feature_frame is None or self._feature_frame.empty:
            return self._hold(ticker, "features unavailable")
        ts = pd.Timestamp(timestamp).tz_localize(None) if timestamp else None
        if ts is None:
            return self._hold(ticker, "invalid timestamp")
        history = self._feature_frame.loc[self._feature_frame.index <= ts]
        if history.empty:
            return self._hold(ticker, "insufficient history")
        row = history.iloc[-1]
        features = {column: float(row.get(column, 0.0)) for column in self._feature_columns}
        price_snapshot = {
            "close": float(price),
            "volume": float(row.get("volume", 0.0)),
        }
        account = AccountSnapshot(
            equity=float(max(portfolio_equity, 0.0)),
            cash=float(max(portfolio_cash, 0.0)),
            buying_power=float(max(portfolio_cash, 0.0)),
            portfolio_value=float(max(portfolio_equity, 0.0)),
        )
        decision = self.engine.decide(
            ticker,
            features=features,
            price_snapshot=price_snapshot,
            account=account,
            position=position,
            open_positions=open_positions,
        )
        return decision

    def on_cycle_end(self) -> None:
        self._feature_frame = None
        self._ticker = None

    def _prepare_features(self, ticker: str) -> None:
        if self._config is None:
            return
        start = self._config.start
        end = self._config.end
        price_df = fetch_price_bars(ticker, start, end, timeframe=self._config.bar_timeframe)
        if price_df.empty:
            self._feature_frame = None
            self._ticker = ticker
            return
        price_df.index = pd.to_datetime(price_df.index).tz_localize(None)
        gdelt_window = collect_gdelt_window(
            ticker,
            start=start.replace(tzinfo=None) if isinstance(start, datetime) else start,
            end=end.replace(tzinfo=None) if isinstance(end, datetime) else end,
            timeline_minutes=max(60, self._config.sentiment_window_minutes),
        )
        feature_frame = self.feature_engineer.build_training_frame(
            ticker,
            price_df,
            gdelt_window,
            include_labels=False,
        )
        if feature_frame.empty:
            self._feature_frame = None
            self._ticker = ticker
            return
        self._feature_columns = self.feature_engineer.feature_columns(feature_frame)
        self._feature_frame = feature_frame
        self._ticker = ticker

    def _hold(self, ticker: str, reason: str) -> TradeDecision:
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
            metadata={}
        )
