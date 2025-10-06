"""FastAPI router for invoking backtests from the dashboard."""

from __future__ import annotations

from datetime import datetime
from typing import Dict

from fastapi import APIRouter, HTTPException

from .config import BacktestConfig

router = APIRouter(prefix="/backtester", tags=["backtester"])


@router.get("/available-bots")
async def get_available_bots() -> Dict:
    """Get list of available bot variants."""
    from .bots.base import BOT_REGISTRY
    
    return {
        "bots": list(BOT_REGISTRY.keys()),
        "count": len(BOT_REGISTRY),
        "default": "traderv5" if "traderv5" in BOT_REGISTRY else "traderv4",
    }


@router.post("/run")
async def run_backtest(payload: Dict) -> Dict:
    try:
        from .runner import BacktestRunner
        raw_workers = payload.get("maxWorkers", payload.get("max_workers"))
        try:
            parsed_workers = int(raw_workers) if raw_workers not in (None, "", "null") else None
        except (TypeError, ValueError) as err:
            raise ValueError(f"invalid maxWorkers value: {raw_workers!r}") from err

        # Determine default bot based on what's available
        from .bots.base import BOT_REGISTRY
        default_bot = "traderv5" if "traderv5" in BOT_REGISTRY else "traderv4"
        
        config = BacktestConfig(
            tickers=[ticker.upper() for ticker in payload["tickers"]],
            start=datetime.fromisoformat(payload["start"]),
            end=datetime.fromisoformat(payload["end"]),
            starting_cash=float(payload.get("startingCash", 100_000)),
            sentiment_window_minutes=int(payload.get("sentimentWindowMinutes", 60)),
            bar_timeframe=str(payload.get("timeframe", "15Min")),
            bot_variants=[variant for variant in payload.get("bots", [default_bot])],
            use_vader=bool(payload.get("useVader", True)),
            use_finbert=bool(payload.get("useFinbert", False)),
            use_keybert=bool(payload.get("useKeybert", False)),
            record_trace=bool(payload.get("includeTrace", False)),
            max_workers=parsed_workers,
        )
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=400, detail=f"Invalid payload: {exc}") from exc

    runner = BacktestRunner(config)
    results = runner.run()
    return {
        "config": {
            "tickers": config.tickers,
            "start": config.start.isoformat(),
            "end": config.end.isoformat(),
            "startingCash": config.starting_cash,
            "timeframe": config.bar_timeframe,
            "bots": config.bot_variants,
            "useVader": config.use_vader,
            "useFinbert": config.use_finbert,
            "useKeybert": config.use_keybert,
            "includeTrace": config.record_trace,
        },
        "results": {
            key: {
                "totalReturn": metrics.total_return,
                "alertPrecision": metrics.alert_precision,
                "winRate": metrics.win_rate,
                "maxDrawdown": metrics.max_drawdown,
                "tradeCount": len(metrics.trades),
                "pnlCurve": metrics.pnl_curve,
                "trades": [trade.__dict__ for trade in metrics.trades],
                "featureSummary": metrics.feature_summary,
                "debug": metrics.debug,
            }
            for key, metrics in results.items()
        },
    }
