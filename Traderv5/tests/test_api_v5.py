from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def build_app_with_router() -> TestClient:
    from Traderv5.api import router as v5_router

    app = FastAPI()
    app.include_router(v5_router)
    return TestClient(app)


def test_gdelt_health_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    client = build_app_with_router()

    class DummyWindow:
        def __init__(self) -> None:
            self.timeline = [(datetime.utcnow(), 0.1)] * 3
            self.articles = [
                {"title": "a"},
                {"title": "b"},
            ]
            self.summaries = [object()]

    # Patch service to avoid network
    from Traderv5 import api as api_module

    def fake_fetch(self, symbol, start, end):  # type: ignore[no-untyped-def]
        return DummyWindow()

    monkeypatch.setattr(
        "Traderv5.services.gdelt.GDELTService.fetch_sentiment_window",
        fake_fetch,
        raising=True,
    )

    resp = client.get("/v5/backtest/health/gdelt?symbol=AAPL&minutes=180")
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload["symbol"] == "AAPL"
    assert payload["timelinePoints"] == 3
    assert payload["articles"] == 2
    assert payload["dailySummaries"] == 1
    assert payload["durationMs"] >= 0


def test_train_run_and_status_reports(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    client = build_app_with_router()

    # Stub Trainer to avoid network and long jobs
    class DummyResult:
        def __init__(self, artifacts: Path) -> None:
            self.estimator_name = "logreg"
            self.roc_auc = 0.6
            self.samples = 100
            self.features = ["f1", "f2"]
            self.model_path = artifacts / "trade_classifier.joblib"
            self.metadata_path = artifacts / "trade_classifier_meta.json"

    class DummyTrainer:
        def __init__(self, cfg: Any, params: Any) -> None:
            self._artifacts = Path("Traderv5/model/artifacts")
            self._artifacts.mkdir(parents=True, exist_ok=True)
            (self._artifacts / "trade_classifier.joblib").write_bytes(b"dummy")
            (self._artifacts / "trade_classifier_meta.json").write_text("{}")

        def collect_dataset(self):  # type: ignore[no-untyped-def]
            idx = pd.date_range("2024-01-01", periods=10, freq="D")
            return pd.DataFrame({"close": range(10)}, index=idx)

        def train(self, dataset):  # type: ignore[no-untyped-def]
            return DummyResult(Path("Traderv5/model/artifacts"))

    monkeypatch.setattr("Traderv5.model.training.Trainer", DummyTrainer, raising=True)

    run = client.post("/v5/backtest/train/run", json={})
    assert run.status_code == 200, run.text
    report = run.json()
    assert report["estimator"] == "logreg"
    assert report["rocAuc"] >= 0
    # Status should expose the persisted report
    status = client.get("/v5/backtest/train/status")
    assert status.status_code == 200
    status_json = status.json()
    assert "report" in status_json
    assert status_json["artifacts"]["report"] is not None


def test_v5_backtest_run_metrics_and_serialization(monkeypatch: pytest.MonkeyPatch) -> None:
    client = build_app_with_router()

    # Patch Yahoo fetcher to return a simple daily frame
    import numpy as np
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    frame = pd.DataFrame(
        {
            "open": np.linspace(100, 104, 5),
            "high": np.linspace(101, 105, 5),
            "low": np.linspace(99, 103, 5),
            "close": np.linspace(100, 104, 5),
            "volume": np.ones(5) * 1_000_000,
        },
        index=idx,
    )

    def fake_fetch(self, ticker, *, range_="1mo", interval="1d", include_prepost=False):  # type: ignore[no-untyped-def]
        return frame

    monkeypatch.setattr(
        "Traderv5.data_sources.YahooFinanceDataFetcher.fetch_price_history",
        fake_fetch,
        raising=True,
    )

    payload = {
        "tickers": ["AAPL"],
        "start": "2024-01-01T00:00",
        "end": "2024-01-10T00:00",
    }
    resp = client.post("/v5/backtest/run", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "summary" in data
    assert "metrics" in data
    assert isinstance(data["trades"], list)
    assert isinstance(data["violations"], list)
    assert isinstance(data.get("equityCurve", []), list)




