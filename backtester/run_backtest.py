"""CLI entry point for running backtests."""

from __future__ import annotations

import argparse
from datetime import datetime
import json

from .config import BacktestConfig
from .runner import BacktestRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TraderV4 backtests")
    parser.add_argument("--tickers", nargs="+", required=True, help="Tickers to backtest")
    parser.add_argument("--start", required=True, help="Start datetime (YYYY-MM-DD HH:MM)")
    parser.add_argument("--end", required=True, help="End datetime (YYYY-MM-DD HH:MM)")
    parser.add_argument("--cash", type=float, default=100_000, help="Starting cash")
    parser.add_argument("--window", type=int, default=60, help="Sentiment window in minutes")
    parser.add_argument("--timeframe", default="15Min", help="Bar timeframe (Alpaca format)")
    parser.add_argument("--bots", nargs="+", default=["traderv4"], help="Bot variants to run")
    parser.add_argument("--finbert", action="store_true", help="Enable FinBERT sentiment (slower)")
    parser.add_argument("--keybert", action="store_true", help="Enable KeyBERT keyword extraction")
    parser.add_argument("--no-vader", action="store_true", help="Disable VADER sentiment")
    parser.add_argument("--trace", action="store_true", help="Record a detailed execution trace")
    parser.add_argument("--workers", type=int, default=0, help="Parallel worker processes (0 = auto)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = BacktestConfig(
        tickers=[ticker.upper() for ticker in args.tickers],
        start=datetime.strptime(args.start, "%Y-%m-%d %H:%M"),
        end=datetime.strptime(args.end, "%Y-%m-%d %H:%M"),
        starting_cash=args.cash,
        sentiment_window_minutes=args.window,
        bar_timeframe=args.timeframe,
        bot_variants=args.bots,
        use_vader=not args.no_vader,
        use_finbert=args.finbert,
        use_keybert=args.keybert,
        record_trace=args.trace,
        max_workers=args.workers or None,
    )
    runner = BacktestRunner(config)
    results = runner.run()
    print(json.dumps({
        key: {
            "totalReturn": metrics.total_return,
            "alertPrecision": metrics.alert_precision,
            "winRate": metrics.win_rate,
            "maxDrawdown": metrics.max_drawdown,
            "tradeCount": len(metrics.trades),
            "pnlCurve": metrics.pnl_curve,
            "featureSummary": metrics.feature_summary,
        }
        for key, metrics in results.items()
    }, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
