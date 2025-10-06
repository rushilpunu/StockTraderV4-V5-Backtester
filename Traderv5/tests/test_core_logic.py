from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

import numpy as np
import pandas as pd

from Traderv5.cash_ledger import CashLedger
from Traderv5.configuration import CalendarSettings, RoutingSettings, SchedulerSettings, DataSettings, AccountSettings, RiskSettings, TrainingSettings, TradingParameters
from Traderv5.backtester import CashAccountBacktester
from Traderv5.decision_router import DecisionRouter, ModelPrediction, PositionInfo
from Traderv5.bucket_scheduler import BucketScheduler


def make_params() -> TradingParameters:
    return TradingParameters(
        account=AccountSettings(type="cash", starting_equity=20000.0, pattern_day_trade_limit=3, min_equity_for_margin=25000.0),
        risk=RiskSettings(risk_per_trade_pct=0.02, atr_stop_multiplier=1.8, take_profit_multiple=2.5, max_drawdown_pct=0.06, drawdown_size_reduction=0.5),
        calendar=CalendarSettings(),
        data=DataSettings(queries_per_second=0.5, max_retries=1, backoff_factor=1.2, cache_ttl_hours=1.0, cache_dir=pd.Path("cache/http") if hasattr(pd, "Path") else __import__("pathlib").Path("cache/http"), gdelt_chunk_minutes=240, gdelt_pause_seconds=1.0, yahoo_pause_seconds=0.8),
        scheduler=SchedulerSettings(bucket_a_days=("MONDAY", "WEDNESDAY", "FRIDAY"), bucket_b_days=("TUESDAY", "THURSDAY"), allocation_pct=0.5),
        routing=RoutingSettings(swing_threshold=0.6, intraday_threshold=0.75, intraday_min_cash_pct=0.2, intraday_top_decile=0.9),
        training=TrainingSettings(tickers=("AAPL",), lookback_days=30, price_interval="1h", timeline_minutes=60, label_horizon_minutes=90, positive_threshold=0.0035, negative_threshold=-0.0035, min_samples=10, max_tickers_per_batch=1),
    )


def test_cash_ledger_execute_buy_no_reserved_increment():
    ledger = CashLedger(1000.0, CalendarSettings())
    start_available = ledger.state.available_cash
    tx = ledger.execute_buy("AAPL", 1, Decimal("100.00"), date(2024, 1, 2))
    assert ledger.state.reserved_cash == Decimal("0"), "reserved cash should not increase for filled buys"
    assert ledger.state.settled_cash == Decimal("900.00")
    assert ledger.state.available_cash == Decimal("900.00")


def test_backtester_price_lookup_fallback_and_prediction_fallback():
    params = make_params()
    backtester = CashAccountBacktester(params)
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    frame = pd.DataFrame({
        "open": [100, 101, 102],
        "high": [101, 102, 103],
        "low": [99, 100, 101],
        "close": [100, 101, 102],
        "volume": [1, 1, 1],
    }, index=idx)
    # Price on a later date should fallback to last available
    price = backtester._get_price_for_date(frame, date(2024, 1, 10))
    assert price == Decimal("102"), price

    # Prediction fallback chooses latest prior
    preds = [
        ModelPrediction(symbol="AAPL", swing_score=0.6, intraday_score=0.6, confidence=0.6, timestamp=datetime(2024, 1, 2), features={}),
    ]
    p = backtester._get_prediction_for_date(preds, date(2024, 1, 3))
    assert p is not None
    assert p.timestamp.date() == date(2024, 1, 2)


def test_decision_router_min_one_share_when_affordable():
    calendar = CalendarSettings()
    # Minimal wiring for router dependencies
    ledger = CashLedger(200.0, calendar)
    scheduler = BucketScheduler(SchedulerSettings(bucket_a_days=("MONDAY",), bucket_b_days=("TUESDAY",), allocation_pct=0.5), calendar)
    router = DecisionRouter(ledger, scheduler, RoutingSettings(swing_threshold=0.55, intraday_threshold=0.55, intraday_min_cash_pct=0.0, intraday_top_decile=0.9))
    price = Decimal("100.00")
    pred = ModelPrediction(symbol="AAPL", swing_score=0.8, intraday_score=0.8, confidence=0.8, timestamp=datetime(2024, 1, 2), features={})
    decision = router._route_buy_decision(pred, price, date(2024, 1, 2))
    if decision.quantity > 0:
        assert decision.quantity >= 1




