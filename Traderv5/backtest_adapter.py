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
from Traderv5.features import FeatureEngineer, FeatureEngineerConfig
from Traderv5.model.predictor import ModelPredictor
from Traderv5.risk_profiles import apply_profile, get_profile
from Traderv5.swing.model import load_swing_predictor


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
        overrides: Dict[str, float] = {}
        try:
            profile = get_profile(getattr(config, "risk_profile", "balanced"))
        except KeyError:
            profile = get_profile("balanced")

        baseline_fraction = max(profile.max_capital_fraction, config.starting_cash / 650_000)
        overrides["max_capital_fraction"] = min(0.45, float(config.max_capital_fraction_override or baseline_fraction))

        if config.max_positions_override is not None:
            overrides["max_positions"] = float(config.max_positions_override)

        if config.cooldown_minutes is not None:
            overrides["cooldown"] = float(config.cooldown_minutes)

        if config.entry_threshold is not None:
            overrides["entry"] = float(config.entry_threshold)

        if config.exit_threshold is not None:
            overrides["exit"] = float(config.exit_threshold)

        if config.aggressiveness is not None:
            overrides["aggressiveness"] = profile.aggressiveness * float(config.aggressiveness)

        if config.entry_signal_bias is not None:
            overrides["bias"] = profile.entry_bias + float(config.entry_signal_bias)

        if config.max_trade_leverage is not None:
            overrides["leverage"] = float(config.max_trade_leverage)

        apply_profile(self.risk, profile, overrides=overrides)
        self.risk.allow_shorting = True
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


@register_bot
class TraderV5SwingBot(BaseBot):
    name = "traderv5_swing"

    def __init__(self) -> None:
        swing_config = FeatureEngineerConfig(
            price_return_windows=(1, 3, 5, 10, 20),
            volatility_windows=(14,),
            atr_window=14,
            moving_average_windows=(10, 20, 30),
            volume_zscore_window=30,
            sentiment_momentum_windows=(1, 3, 7),
            sentiment_zscore_window=14,
            label_horizon_minutes=60 * 24 * 5,
            positive_threshold=0.006,
            negative_threshold=-0.006,
        )
        self.feature_engineer = FeatureEngineer(swing_config)
        try:
            self.predictor = load_swing_predictor()
        except Exception as exc:  # pragma: no cover - defensive guard for missing deps
            raise RuntimeError(f"Failed to initialise swing predictor: {exc}") from exc
        self.risk = RiskConfig()
        self.trade_tracker = TradeFrequencyTracker(max_day_trades=3)
        self.engine = ModelDecisionEngine(self.predictor, self.risk, self.trade_tracker)
        self._feature_frame: Optional[pd.DataFrame] = None
        self._feature_columns: list[str] = []
        self._ticker: Optional[str] = None
        self._config: Optional[BacktestConfig] = None

    def setup(self, config: BacktestConfig, baseline: Dict[str, float]) -> None:
        self._config = config
        try:
            profile = get_profile("swing")
        except KeyError:
            profile = get_profile("balanced")

        overrides: Dict[str, float] = {
            "max_capital_fraction": min(0.5, float(config.max_capital_fraction_override or profile.max_capital_fraction)),
            "max_positions": float(config.max_positions_override or profile.max_positions),
            "cooldown": float(config.cooldown_minutes or profile.cooldown),
        }
        if config.entry_threshold is not None:
            overrides["entry"] = float(config.entry_threshold)
        if config.exit_threshold is not None:
            overrides["exit"] = float(config.exit_threshold)
        if config.aggressiveness is not None:
            overrides["aggressiveness"] = profile.aggressiveness * float(config.aggressiveness)
        if config.entry_signal_bias is not None:
            overrides["bias"] = profile.entry_bias + float(config.entry_signal_bias)
        if config.max_trade_leverage is not None:
            overrides["leverage"] = float(config.max_trade_leverage)

        apply_profile(self.risk, profile, overrides=overrides)
        self.risk.allow_shorting = False

        self.trade_tracker = TradeFrequencyTracker(max_day_trades=3)
        self.engine = ModelDecisionEngine(self.predictor, self.risk, self.trade_tracker)
        self._feature_frame = None
        self._ticker = None
        self._feature_columns = []

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
        if not self._feature_columns:
            self._feature_columns = [col for col in row.index if col not in {"label", "forward_return", "ticker"}]
        features = {column: float(row.get(column, 0.0)) for column in self._feature_columns}

        price_snapshot = {
            "close": float(price),
            "volume": float(row.get("volume", 0.0)),
        }
        account = AccountSnapshot(
            equity=float(max(portfolio_equity, 0.0)),
            cash=float(max(portfolio_cash, 0.0)),
            buying_power=float(max(portfolio_cash, portfolio_equity * getattr(self.risk, "max_trade_leverage", 1.0))),
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
        self._feature_columns = []

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
        # Resample to daily bars to align with swing cadence.
        daily = price_df.resample("1D").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna(how="all")
        daily = daily.ffill()
        if daily.empty:
            self._feature_frame = None
            self._ticker = ticker
            return

        feature_frame = self.feature_engineer.build_training_frame(
            ticker,
            daily,
            gdelt_window=None,
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
            metadata={},
        )
