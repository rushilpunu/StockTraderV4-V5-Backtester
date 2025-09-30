"""FastAPI application exposing TraderV4 state for dashboards."""

from __future__ import annotations

import asyncio
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from Traderv4.main import (
    NewsSentimentTrader,
    TraderConfig,
    default_watchlist,
    _load_account_balance,
    _load_alpaca_client,
)
from backtester.api import router as backtester_router
from Traderv4.state import StateStore


def create_app(trader: NewsSentimentTrader, state_store: StateStore) -> FastAPI:
    app = FastAPI(title="TraderV4 API", version="0.1.0")
    store = state_store

    allow_origins = [
        "http://localhost",
        "http://localhost:19006",
        "http://127.0.0.1:19006",
        "http://localhost:8081",
    ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(backtester_router)

    @app.get("/")
    async def index() -> dict:
        return {
            "message": "TraderV4 API running",
            "endpoints": [
                "/health",
                "/snapshot",
                "/alerts",
                "/positions",
                "/day-trade",
                "/sentiment",
                "/account",
                "/settings",
                "/run",
                "/docs",
            ],
        }

    @app.get("/health")
    async def health_check() -> dict:
        return {"status": "ok"}

    @app.get("/snapshot")
    async def snapshot() -> dict:
        return store.snapshot().as_dict()

    @app.get("/alerts")
    async def alerts() -> dict:
        snap = store.snapshot()
        return {"alerts": [alert.as_dict() for alert in snap.alerts], "lastUpdated": snap.last_updated.isoformat() if snap.last_updated else None}

    @app.get("/positions")
    async def positions() -> dict:
        snap = store.snapshot()
        return {
            "positions": [position.as_dict() for position in snap.positions],
            "lastUpdated": snap.last_updated.isoformat() if snap.last_updated else None,
        }

    @app.get("/day-trade")
    async def day_trade() -> dict:
        snap = store.snapshot()
        return {
            "dayTrade": snap.day_trade.as_dict() if snap.day_trade else None,
            "lastUpdated": snap.last_updated.isoformat() if snap.last_updated else None,
        }

    @app.get("/sentiment")
    async def sentiment_history() -> dict:
        snap = store.snapshot()
        return {
            "history": [point.as_dict() for point in snap.sentiment_history],
            "lastUpdated": snap.last_updated.isoformat() if snap.last_updated else None,
        }

    @app.get("/diagnostics")
    async def diagnostics() -> dict:
        snap = store.snapshot()
        diag = snap.diagnostics
        return {
            "diagnostics": diag.as_dict() if diag else None,
            "lastUpdated": snap.last_updated.isoformat() if snap.last_updated else None,
        }

    @app.get("/settings")
    async def get_settings() -> dict:
        return trader.get_settings()

    @app.post("/settings")
    async def update_settings(payload: dict) -> dict:
        return trader.update_settings(payload)

    @app.get("/account")
    async def account_summary() -> dict:
        try:
            account = trader.executor.alpaca.get_account()
        except Exception as exc:  # pragma: no cover - runtime safeguard
            raise HTTPException(status_code=502, detail=f"Failed to fetch account: {exc}") from exc

        return {
            "id": getattr(account, "id", ""),
            "status": getattr(account, "status", ""),
            "currency": getattr(account, "currency", "USD"),
            "cash": float(getattr(account, "cash", 0.0)),
            "buyingPower": float(getattr(account, "buying_power", 0.0)),
            "portfolioValue": float(getattr(account, "portfolio_value", 0.0)),
            "equity": float(getattr(account, "equity", 0.0)),
            "daytradeCount": getattr(account, "daytrade_count", 0),
            "createdAt": getattr(account, "created_at", None),
            "multiplier": getattr(account, "multiplier", None),
        }

    @app.get("/account/positions")
    async def account_positions() -> dict:
        positions_payload = trader.executor.get_open_positions()
        return {"positions": positions_payload}

    @app.post("/run")
    async def run_cycle() -> dict:
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, trader.run_cycle)
        except Exception as exc:  # pragma: no cover - safeguard for runtime errors
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return store.snapshot().as_dict()

    return app


def build_trader(config: Optional[TraderConfig] = None) -> NewsSentimentTrader:
    if config is None:
        config = TraderConfig(tickers=default_watchlist())
    alpaca = _load_alpaca_client()
    state_store = StateStore()
    trader = NewsSentimentTrader(
        config=config,
        alpaca_client=alpaca,
        balance_fetcher=lambda: _load_account_balance(alpaca),
        state_store=state_store,
    )
    return trader


__all__ = ["create_app", "build_trader"]
