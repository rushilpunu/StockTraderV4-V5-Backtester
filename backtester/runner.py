"""Backtest orchestration."""

from __future__ import annotations

import concurrent.futures
import logging
import os
from collections import Counter, deque
from datetime import datetime
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from .bots.base import BOT_REGISTRY, BaseBot
from .config import BacktestConfig
from .data_sources import fetch_price_bars
from .metrics import BacktestMetrics
from .sentiment_features import SentimentSnapshot, build_sentiment_snapshots
from .simulator import ExecutedTrade, PortfolioSimulator
from Traderv4.funcs import TradeDecision

_LOG = logging.getLogger(__name__)


class BacktestRunner:
    def __init__(self, config: BacktestConfig) -> None:
        self.config = config
        self.bot_names = list(config.bot_variants)

    def run(self) -> Dict[str, BacktestMetrics]:
        tickers = list(dict.fromkeys(self.config.tickers))
        if not tickers:
            return {}

        requested_workers = self.config.max_workers or min(len(tickers), os.cpu_count() or 1)
        if requested_workers > 1 and len(tickers) > 1:
            return self._run_parallel(tickers, requested_workers)
        return self._run_sequential(tickers)

    def _run_sequential(self, tickers: Iterable[str]) -> Dict[str, BacktestMetrics]:
        results: Dict[str, BacktestMetrics] = {}
        for ticker in tickers:
            ticker_results = self._run_ticker(self.config, self.bot_names, ticker)
            results.update(ticker_results)
        return results

    def _run_parallel(self, tickers: Iterable[str], requested_workers: int) -> Dict[str, BacktestMetrics]:
        ticker_list = list(tickers)
        workers = max(1, min(requested_workers, len(ticker_list)))
        if workers <= 1:
            return self._run_sequential(ticker_list)
        results: Dict[str, BacktestMetrics] = {}
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_run_ticker_job, self.config, self.bot_names, ticker): ticker
                for ticker in ticker_list
            }
            for future in concurrent.futures.as_completed(futures):
                ticker = futures[future]
                try:
                    ticker_results = future.result()
                except Exception as exc:  # pragma: no cover - propagated to logs
                    _LOG.error("Backtest failed for %s: %s", ticker, exc, exc_info=True)
                    continue
                results.update(ticker_results)
        return results

    @staticmethod
    def _prepare_sentiment(config: BacktestConfig, ticker: str) -> List[Tuple[pd.Timestamp, SentimentSnapshot]]:
        series = build_sentiment_snapshots(
            ticker,
            config.start,
            config.end,
            granularity_minutes=config.sentiment_window_minutes,
            use_vader=config.use_vader,
            use_finbert=config.use_finbert,
            use_keybert=config.use_keybert,
        )
        sentiment_pairs: List[Tuple[pd.Timestamp, SentimentSnapshot]] = [
            (pd.Timestamp(point.timestamp).tz_localize(None), point) for point in series
        ]
        sentiment_pairs.sort(key=lambda pair: pair[0])
        return sentiment_pairs

    @staticmethod
    def _compute_baseline(price_df: pd.DataFrame) -> Dict[str, float]:
        high = price_df["high"] if "high" in price_df else price_df["h"] if "h" in price_df else None
        low = price_df["low"] if "low" in price_df else price_df["l"] if "l" in price_df else None
        close = price_df["close"] if "close" in price_df else price_df["c"] if "c" in price_df else None
        volume = price_df["volume"] if "volume" in price_df else price_df["v"] if "v" in price_df else None

        avg_range_pct = 0.0
        if high is not None and low is not None and close is not None:
            ranges = ((high - low) / close).abs()
            avg_range_pct = float(ranges.mean()) if not ranges.empty else 0.0

        avg_volume = float(volume.mean()) if volume is not None and not volume.empty else 0.0
        return {"avg_range_pct": avg_range_pct, "avg_volume": avg_volume}

    @staticmethod
    def _instantiate_bot(name: str) -> BaseBot:
        template = BOT_REGISTRY.get(name)
        if template is None:
            raise KeyError(f"Unknown bot: {name}")
        return template.__class__()

    @classmethod
    def _run_ticker(cls, config: BacktestConfig, bot_names: Iterable[str], ticker: str) -> Dict[str, BacktestMetrics]:
        results: Dict[str, BacktestMetrics] = {}
        price_df = fetch_price_bars(ticker, config.start, config.end, timeframe=config.bar_timeframe)
        if price_df.empty:
            _LOG.warning("Skipping %s: no price data", ticker)
            return results
        sentiment_pairs = cls._prepare_sentiment(config, ticker)
        if not sentiment_pairs:
            _LOG.warning("No sentiment snapshots for %s in selected range", ticker)
            return results

        baseline = cls._compute_baseline(price_df)
        close_col = "close" if "close" in price_df.columns else "c" if "c" in price_df.columns else None
        if close_col is None:
            _LOG.warning("Price data for %s missing close column", ticker)
            return results

        for bot_name in bot_names:
            bot = cls._instantiate_bot(bot_name)
            simulator = PortfolioSimulator(config.starting_cash)
            alerts = 0
            sentiment_idx = 0
            step_logs: List[Dict[str, Any]] = [] if config.record_trace else None
            trade_scores: List[float] = []
            keyword_counter: Counter[str] = Counter()
            bot.setup(config, baseline)
            for row in price_df.itertuples(index=True):
                ts = row.Index
                timestamp = pd.Timestamp(ts).tz_localize(None) if isinstance(ts, pd.Timestamp) else pd.Timestamp(ts)
                price = float(getattr(row, close_col))
                snapshot = None
                while sentiment_idx < len(sentiment_pairs) and sentiment_pairs[sentiment_idx][0] <= timestamp:
                    snapshot = sentiment_pairs[sentiment_idx][1]
                    sentiment_idx += 1
                if snapshot is None:
                    simulator.step(timestamp.isoformat(), {ticker: price})
                    continue
                equity_before = simulator.equity_curve[-1] if simulator.equity_curve else simulator.starting_cash
                position_quantity = simulator.positions.get(ticker, 0.0)
                open_positions = sum(1 for shares in simulator.positions.values() if abs(shares) > 1e-6)
                position_ctx = None
                if abs(position_quantity) > 1e-6:
                    side = "LONG" if position_quantity > 0 else "SHORT"
                    position_ctx = {
                        "ticker": ticker,
                        "side": side,
                        "quantity": position_quantity,
                        "market_value": position_quantity * price,
                    }
                decision = bot.on_snapshot(
                    timestamp=timestamp.isoformat(),
                    ticker=ticker,
                    price=price,
                    sentiment_snapshot=snapshot,
                    portfolio_cash=simulator.cash,
                    portfolio_equity=equity_before,
                    baseline=baseline,
                    position=position_ctx,
                    open_positions=open_positions,
                )
                if decision.action != "HOLD":
                    pre_equity = simulator.equity_curve[-1] if simulator.equity_curve else simulator.starting_cash
                    shares = simulator.execute(decision, price, timestamp.isoformat())
                    if shares > 0:
                        bot.on_trade_filled(
                            decision=decision,
                            timestamp=timestamp.isoformat(),
                            price=price,
                            quantity=shares,
                        )
                    simulator.step(timestamp.isoformat(), {ticker: price})
                    post_equity = simulator.equity_curve[-1] if simulator.equity_curve else pre_equity
                    if shares > 0:
                        alerts += 1
                        score_val = decision.metadata.get("score") if hasattr(decision, "metadata") else None
                        if score_val is not None:
                            try:
                                trade_scores.append(float(score_val))
                            except (TypeError, ValueError):
                                pass
                        keywords_val = decision.metadata.get("keywords") if hasattr(decision, "metadata") else ""
                        if keywords_val:
                            for kw in keywords_val.split(","):
                                if kw:
                                    keyword_counter[kw.strip()] += 1
                else:
                    simulator.step(timestamp.isoformat(), {ticker: price})
                if step_logs is not None and len(step_logs) < 250:
                    step_logs.append(
                        {
                            "timestamp": timestamp.isoformat(),
                            "price": price,
                            "tone15": getattr(snapshot, "tone_15", 0.0),
                            "delta15": getattr(snapshot, "delta_15", 0.0),
                            "score": decision.metadata.get("score") if hasattr(decision, "metadata") else None,
                            "action": decision.action,
                            "notional": decision.notional,
                            "cash": simulator.cash,
                            "equity": simulator.equity_curve[-1] if simulator.equity_curve else simulator.starting_cash,
                        }
                    )
            last_price = float(price_df[close_col].iloc[-1])
            final_timestamp = price_df.index[-1]
            final_ts = pd.Timestamp(final_timestamp).tz_localize(None) if isinstance(final_timestamp, pd.Timestamp) else pd.Timestamp(final_timestamp)
            forced_trades = simulator.finalize({ticker: last_price}, timestamp=final_ts.isoformat())
            for trade in forced_trades:
                forced_decision = TradeDecision(
                    ticker=trade.ticker,
                    action=trade.action,
                    confidence=0.0,
                    notional=trade.notional,
                    time_in_force="gtc",
                    stop_loss=None,
                    take_profit=None,
                    reason=trade.reason,
                    intent="exit",
                    quantity=trade.quantity,
                    metadata={},
                )
                bot.on_trade_filled(
                    decision=forced_decision,
                    timestamp=trade.timestamp,
                    price=trade.price,
                    quantity=trade.quantity,
                )
            results_key = f"{bot_name}:{ticker}"
            feature_summary = {
                "averageScore": float(np.mean(trade_scores)) if trade_scores else 0.0,
                "tradeCount": len(simulator.trade_log),
                "topKeywords": [kw for kw, _ in keyword_counter.most_common(5)],
                "snapshots": len(sentiment_pairs),
            }
            results[results_key] = BacktestMetrics(
                total_return=simulator.realized_pnl(),
                trades=simulator.trade_log,
                pnl_curve=simulator.equity_curve,
                profitable_trades=_count_profitable(simulator.trade_log),
                total_alerts=alerts,
                debug={
                    "steps": step_logs or [],
                    "sentimentSnapshots": len(sentiment_pairs),
                    "pricePoints": len(price_df),
                },
                feature_summary=feature_summary,
            )
            bot.on_cycle_end()
        return results


def _count_profitable(trades: Iterable[ExecutedTrade]) -> int:
    """Pair buy/sell executions to count profitable exits."""

    long_entries: deque[list[float]] = deque()
    wins = 0
    for trade in trades:
        if trade.action == "BUY":
            long_entries.append([float(trade.price), float(trade.quantity)])
            continue
        if trade.action != "SELL":
            continue
        remaining = float(trade.quantity)
        while remaining > 1e-6 and long_entries:
            entry_price, entry_qty = long_entries[0]
            matched = min(entry_qty, remaining)
            pnl = (float(trade.price) - entry_price) * matched
            if pnl > 0:
                wins += 1
            entry_qty -= matched
            remaining -= matched
            if entry_qty <= 1e-6:
                long_entries.popleft()
            else:
                long_entries[0][1] = entry_qty
    return wins


def _run_ticker_job(config: BacktestConfig, bot_names: Iterable[str], ticker: str) -> Dict[str, BacktestMetrics]:
    return BacktestRunner._run_ticker(config, bot_names, ticker)


__all__ = ["BacktestRunner"]
