"""Configuration loading and validation utilities for Trader V5."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "trader_v5.json"
_ALLOWED_ACCOUNT_TYPES = {"cash", "margin"}


def _ensure_path(path: Optional[str | Path]) -> Path:
    if path is None:
        return _DEFAULT_CONFIG_PATH
    return Path(path).expanduser().resolve()


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _parse_date(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Invalid date format '{value}'; expected YYYY-MM-DD") from exc


@dataclass(frozen=True)
class AccountSettings:
    type: str
    starting_equity: float
    pattern_day_trade_limit: int
    min_equity_for_margin: float

    def __post_init__(self) -> None:
        account_type = self.type.lower()
        if account_type not in _ALLOWED_ACCOUNT_TYPES:
            raise ValueError(f"account.type must be one of {_ALLOWED_ACCOUNT_TYPES}, got '{self.type}'")
        if self.starting_equity <= 0:
            raise ValueError("account.starting_equity must be positive")
        if self.pattern_day_trade_limit <= 0:
            raise ValueError("account.pattern_day_trade_limit must be positive")
        if self.min_equity_for_margin < 0:
            raise ValueError("account.min_equity_for_margin must be non-negative")

    @property
    def is_cash(self) -> bool:
        return self.type.lower() == "cash"


@dataclass(frozen=True)
class RiskSettings:
    risk_per_trade_pct: float
    atr_stop_multiplier: float
    take_profit_multiple: float
    max_drawdown_pct: float
    drawdown_size_reduction: float
    entry_signal_bias: float = 0.0
    aggressiveness: float = 1.0
    max_trade_leverage: float = 1.0

    def __post_init__(self) -> None:
        if not (0.0 < self.risk_per_trade_pct < 0.05):
            raise ValueError("risk.risk_per_trade_pct must be > 0 and < 0.05")
        if self.atr_stop_multiplier <= 0:
            raise ValueError("risk.atr_stop_multiplier must be positive")
        if self.take_profit_multiple <= 0:
            raise ValueError("risk.take_profit_multiple must be positive")
        if not (0.0 < self.max_drawdown_pct < 0.5):
            raise ValueError("risk.max_drawdown_pct must be between 0 and 0.5")
        if not (0.0 < self.drawdown_size_reduction <= 1.0):
            raise ValueError("risk.drawdown_size_reduction must be within (0, 1]")
        if self.entry_signal_bias < 0:
            raise ValueError("risk.entry_signal_bias must be non-negative")
        if self.aggressiveness <= 0:
            raise ValueError("risk.aggressiveness must be positive")
        if self.max_trade_leverage <= 0:
            raise ValueError("risk.max_trade_leverage must be positive")


@dataclass(frozen=True)
class CalendarSettings:
    exchange: str = "XNYS"
    holiday_calendar: str = "XNYS"
    custom_holidays: Sequence[date] = field(default_factory=tuple)
    settlement_lag_days: int = 1

    def __post_init__(self) -> None:
        if self.settlement_lag_days <= 0:
            raise ValueError("calendar.settlement_lag_days must be positive")


@dataclass(frozen=True)
class DataSettings:
    queries_per_second: float
    max_retries: int
    backoff_factor: float
    cache_ttl_hours: float
    cache_dir: Path
    gdelt_chunk_minutes: int
    gdelt_pause_seconds: float
    yahoo_pause_seconds: float

    def __post_init__(self) -> None:
        if self.queries_per_second <= 0:
            raise ValueError("data.queries_per_second must be positive")
        if self.max_retries < 0:
            raise ValueError("data.max_retries must be non-negative")
        if self.backoff_factor <= 0:
            raise ValueError("data.backoff_factor must be positive")
        if self.cache_ttl_hours <= 0:
            raise ValueError("data.cache_ttl_hours must be positive")
        if self.gdelt_chunk_minutes <= 0:
            raise ValueError("data.gdelt_chunk_minutes must be positive")
        if self.gdelt_pause_seconds < 0:
            raise ValueError("data.gdelt_pause_seconds must be non-negative")
        if self.yahoo_pause_seconds < 0:
            raise ValueError("data.yahoo_pause_seconds must be non-negative")


@dataclass(frozen=True)
class SchedulerSettings:
    bucket_a_days: Sequence[str]
    bucket_b_days: Sequence[str]
    allocation_pct: float

    def __post_init__(self) -> None:
        if not self.bucket_a_days or not self.bucket_b_days:
            raise ValueError("scheduler bucket day assignments must be non-empty")
        normalized_a = {day.upper() for day in self.bucket_a_days}
        normalized_b = {day.upper() for day in self.bucket_b_days}
        valid_days = {
            "MONDAY",
            "TUESDAY",
            "WEDNESDAY",
            "THURSDAY",
            "FRIDAY",
        }
        if not normalized_a <= valid_days:
            raise ValueError("scheduler.bucket_a_days contains invalid weekday names")
        if not normalized_b <= valid_days:
            raise ValueError("scheduler.bucket_b_days contains invalid weekday names")
        if normalized_a & normalized_b:
            raise ValueError("scheduler buckets must not share trading days")
        if not (0.0 < self.allocation_pct < 1.0):
            raise ValueError("scheduler.allocation_pct must be between 0 and 1 (exclusive)")


@dataclass(frozen=True)
class RoutingSettings:
    swing_threshold: float
    intraday_threshold: float
    intraday_min_cash_pct: float
    intraday_top_decile: float
    hold_if_unsettled_needed: bool = True

    def __post_init__(self) -> None:
        if not (0.0 < self.swing_threshold < 1.0):
            raise ValueError("routing.swing_threshold must be between 0 and 1")
        if not (0.0 < self.intraday_threshold < 1.0):
            raise ValueError("routing.intraday_threshold must be between 0 and 1")
        if not (0.0 < self.intraday_min_cash_pct <= 1.0):
            raise ValueError("routing.intraday_min_cash_pct must be within (0, 1]")
        if not (0.0 < self.intraday_top_decile <= 1.0):
            raise ValueError("routing.intraday_top_decile must be within (0, 1]")


@dataclass(frozen=True)
class TrainingSettings:
    tickers: Sequence[str]
    lookback_days: int
    price_interval: str
    timeline_minutes: int
    label_horizon_minutes: int
    positive_threshold: float
    negative_threshold: float
    min_samples: int
    max_tickers_per_batch: int

    def __post_init__(self) -> None:
        if not self.tickers:
            raise ValueError("training.tickers must be non-empty")
        if self.lookback_days <= 0:
            raise ValueError("training.lookback_days must be positive")
        if self.timeline_minutes <= 0:
            raise ValueError("training.timeline_minutes must be positive")
        if self.label_horizon_minutes <= 0:
            raise ValueError("training.label_horizon_minutes must be positive")
        if self.min_samples <= 0:
            raise ValueError("training.min_samples must be positive")
        if self.max_tickers_per_batch <= 0:
            raise ValueError("training.max_tickers_per_batch must be positive")
        if not (-1.0 < self.negative_threshold < 0.0):
            raise ValueError("training.negative_threshold must be between -1 and 0")
        if not (0.0 < self.positive_threshold < 1.0):
            raise ValueError("training.positive_threshold must be between 0 and 1")


@dataclass(frozen=True)
class TradingParameters:
    account: AccountSettings
    risk: RiskSettings
    calendar: CalendarSettings
    data: DataSettings
    scheduler: SchedulerSettings
    routing: RoutingSettings
    training: TrainingSettings

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _coerce_custom_holidays(values: Iterable[Any]) -> List[date]:
    holidays: List[date] = []
    for raw in values:
        if isinstance(raw, date):
            holidays.append(raw)
        elif isinstance(raw, str):
            holidays.append(_parse_date(raw))
        else:
            raise ValueError("calendar.custom_holidays entries must be date objects or YYYY-MM-DD strings")
    return holidays


def _build_settings(raw: Mapping[str, Any]) -> TradingParameters:
    account_raw = raw.get("account", {})
    risk_raw = raw.get("risk", {})
    calendar_raw = raw.get("calendar", {})
    data_raw = raw.get("data", {})
    scheduler_raw = raw.get("scheduler", {})
    routing_raw = raw.get("routing", {})
    training_raw = raw.get("training", {})

    account = AccountSettings(
        type=account_raw.get("type", "cash"),
        starting_equity=float(account_raw.get("starting_equity", 25000.0)),
        pattern_day_trade_limit=int(account_raw.get("pattern_day_trade_limit", 3)),
        min_equity_for_margin=float(account_raw.get("min_equity_for_margin", 25000.0)),
    )

    risk = RiskSettings(
        risk_per_trade_pct=float(risk_raw.get("risk_per_trade_pct", 0.02)),
        atr_stop_multiplier=float(risk_raw.get("atr_stop_multiplier", 1.8)),
        take_profit_multiple=float(risk_raw.get("take_profit_multiple", 2.5)),
        max_drawdown_pct=float(risk_raw.get("max_drawdown_pct", 0.06)),
        drawdown_size_reduction=float(risk_raw.get("drawdown_size_reduction", 0.5)),
        entry_signal_bias=float(risk_raw.get("entry_signal_bias", 0.0)),
        aggressiveness=float(risk_raw.get("aggressiveness", 1.0)),
        max_trade_leverage=float(risk_raw.get("max_trade_leverage", 1.0)),
    )

    calendar = CalendarSettings(
        exchange=str(calendar_raw.get("exchange", "XNYS")),
        holiday_calendar=str(calendar_raw.get("holiday_calendar", "XNYS")),
        custom_holidays=tuple(_coerce_custom_holidays(calendar_raw.get("custom_holidays", []))),
        settlement_lag_days=int(calendar_raw.get("settlement_lag_days", 1)),
    )

    data = DataSettings(
        queries_per_second=float(data_raw.get("queries_per_second", 0.5)),
        max_retries=int(data_raw.get("max_retries", 4)),
        backoff_factor=float(data_raw.get("backoff_factor", 1.8)),
        cache_ttl_hours=float(data_raw.get("cache_ttl_hours", 24)),
        cache_dir=Path(data_raw.get("cache_dir", "cache/http")).expanduser(),
        gdelt_chunk_minutes=int(data_raw.get("gdelt_chunk_minutes", 240)),
        gdelt_pause_seconds=float(data_raw.get("gdelt_pause_seconds", 1.0)),
        yahoo_pause_seconds=float(data_raw.get("yahoo_pause_seconds", 0.8)),
    )

    scheduler = SchedulerSettings(
        bucket_a_days=tuple(scheduler_raw.get("bucket_a_days", ("MONDAY", "WEDNESDAY", "FRIDAY"))),
        bucket_b_days=tuple(scheduler_raw.get("bucket_b_days", ("TUESDAY", "THURSDAY"))),
        allocation_pct=float(scheduler_raw.get("allocation_pct", 0.5)),
    )

    routing = RoutingSettings(
        swing_threshold=float(routing_raw.get("swing_threshold", 0.6)),
        intraday_threshold=float(routing_raw.get("intraday_threshold", 0.75)),
        intraday_min_cash_pct=float(routing_raw.get("intraday_min_cash_pct", 0.2)),
        intraday_top_decile=float(routing_raw.get("intraday_top_decile", 0.9)),
        hold_if_unsettled_needed=bool(routing_raw.get("hold_if_unsettled_needed", True)),
    )

    training = TrainingSettings(
        tickers=tuple(str(t).upper() for t in training_raw.get("tickers", ("AAPL", "MSFT", "AMZN"))),
        lookback_days=int(training_raw.get("lookback_days", 120)),
        price_interval=str(training_raw.get("price_interval", "1h")),
        timeline_minutes=int(training_raw.get("timeline_minutes", 60)),
        label_horizon_minutes=int(training_raw.get("label_horizon_minutes", 90)),
        positive_threshold=float(training_raw.get("positive_threshold", 0.0035)),
        negative_threshold=float(training_raw.get("negative_threshold", -0.0035)),
        min_samples=int(training_raw.get("min_samples", 200)),
        max_tickers_per_batch=int(training_raw.get("max_tickers_per_batch", 2)),
    )

    return TradingParameters(
        account=account,
        risk=risk,
        calendar=calendar,
        data=data,
        scheduler=scheduler,
        routing=routing,
        training=training,
    )


def load_trading_config(path: Optional[str | Path] = None) -> Dict[str, Any]:
    """Load and validate trading parameters from configuration.

    Parameters
    ----------
    path:
        Optional override path for the configuration file. When omitted,
        ``config/trader_v5.json`` relative to the project root is used.

    Returns
    -------
    dict
        Nested dictionary of validated configuration suitable for consumption
        by other modules.
    """

    config_path = _ensure_path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Trading configuration file not found at {config_path}")
    raw = _load_json(config_path)
    settings = _build_settings(raw)
    result = settings.to_dict()
    result["config_path"] = str(config_path)
    return result


def load_trading_parameters(path: Optional[str | Path] = None) -> TradingParameters:
    """Load and validate trading parameters as TradingParameters object.

    Parameters
    ----------
    path:
        Optional override path for the configuration file. When omitted,
        ``config/trader_v5.json`` relative to the project root is used.

    Returns
    -------
    TradingParameters
        Validated trading parameters object.
    """
    config_path = _ensure_path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Trading configuration file not found at {config_path}")
    raw = _load_json(config_path)
    return _build_settings(raw)


__all__ = [
    "load_trading_config",
    "load_trading_parameters",
    "TradingParameters",
    "AccountSettings",
    "RiskSettings",
    "CalendarSettings",
    "DataSettings",
    "SchedulerSettings",
    "RoutingSettings",
    "TrainingSettings",
]
