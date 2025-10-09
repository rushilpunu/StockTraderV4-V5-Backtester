"""Market data and sentiment loaders for backtests."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:  # pragma: no cover - import guard exercised in tests
    from alpaca_trade_api import REST  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    REST = None  # type: ignore[misc]

from Traderv4.GDELT import GDELTClient
from Traderv5.data_sources import YahooFinanceDataFetcher

try:  # pragma: no cover - configuration may be absent in CI
    from config.credentials import load_alpaca_credentials
except Exception:  # pragma: no cover - fallback when config unavailable
    load_alpaca_credentials = None  # type: ignore

_LOG = logging.getLogger(__name__)
_OFFLINE_MODE = os.getenv("BACKTEST_OFFLINE", "1").lower() not in {"0", "false", "no"}

_LOCAL_PRICE_FILE = Path(__file__).resolve().parents[1] / "data" / "prices" / "all_stocks_5yr_subset.csv"
_LOCAL_PRICE_CACHE: Dict[str, pd.DataFrame] = {}


def _alpaca_available() -> bool:
    return REST is not None and load_alpaca_credentials is not None


def _alpaca_rest() -> REST:
    if not _alpaca_available():  # pragma: no cover - defensive guard
        raise RuntimeError("Alpaca credentials unavailable")
    creds = load_alpaca_credentials(required=False) if load_alpaca_credentials else None
    if creds is None:
        raise RuntimeError("Alpaca credentials unavailable")
    return REST(key_id=creds.api_key, secret_key=creds.api_secret, base_url=creds.base_url)


def _format_timestamp(value: datetime) -> str:
    if value.tzinfo is None:
        return value.strftime("%Y-%m-%dT%H:%M:%SZ")
    return value.astimezone().strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_local_price_series(ticker: str) -> pd.DataFrame:
    """Load locally cached OHLCV data for the requested ticker."""

    symbol = ticker.upper()
    cached = _LOCAL_PRICE_CACHE.get(symbol)
    if cached is not None:
        return cached.copy()

    if not _LOCAL_PRICE_FILE.exists():
        return pd.DataFrame()

    frame = pd.read_csv(_LOCAL_PRICE_FILE)
    if "date" not in frame.columns or "Name" not in frame.columns:
        _LOG.warning("Local price cache is missing expected columns")
        return pd.DataFrame()
    frame["date"] = pd.to_datetime(frame["date"])
    subset = frame[frame["Name"].str.upper() == symbol].copy()
    if subset.empty:
        return pd.DataFrame()

    subset = subset.sort_values("date").set_index("date")
    renamed = subset.rename(
        columns={
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
        }
    )
    renamed = renamed[["open", "high", "low", "close", "volume"]]
    renamed = renamed.apply(pd.to_numeric, errors="coerce")
    renamed = renamed.ffill().bfill()
    _LOCAL_PRICE_CACHE[symbol] = renamed
    return renamed.copy()


def _slice_price_frame(df: pd.DataFrame, start: datetime, end: datetime, *, ticker: Optional[str] = None) -> pd.DataFrame:
    if df.empty:
        return df
    start_norm = pd.Timestamp(start).normalize()
    end_norm = pd.Timestamp(end).normalize()
    window = df[(df.index >= start_norm) & (df.index <= end_norm)].copy()
    if window.empty:
        symbol = ticker or "<unknown>"
        _LOG.warning("Local price cache has no data for %s between %s and %s", symbol, start, end)
    return window


def fetch_price_bars(
    ticker: str,
    start: datetime,
    end: datetime,
    timeframe: str = "15Min",
) -> pd.DataFrame:
    """Fetch historical bars from Alpaca market data."""
    if _alpaca_available() and not _OFFLINE_MODE:
        try:
            rest = _alpaca_rest()
            bars = rest.get_bars(
                symbol=ticker,
                timeframe=timeframe,
                start=_format_timestamp(start),
                end=_format_timestamp(end),
            )
            if hasattr(bars, "df"):
                df = bars.df
            else:
                df = pd.DataFrame([bar._raw for bar in bars])  # type: ignore[attr-defined]
            if not df.empty:
                return df.sort_index()
            _LOG.warning("No Alpaca price data returned for %s; falling back to Yahoo", ticker)
        except Exception as exc:  # pragma: no cover - runtime dependency failures
            _LOG.warning("Alpaca data fetch failed for %s (%s); falling back to Yahoo", ticker, exc)

    if _OFFLINE_MODE:
        local = _slice_price_frame(_load_local_price_series(ticker), start, end, ticker=ticker)
        if local.empty:
            raise RuntimeError(
                f"Offline mode enabled but no local price data available for {ticker}. "
                "Please populate data/prices/all_stocks_5yr_subset.csv with real OHLCV prices."
            )
        if timeframe != "1Day":
            _LOG.warning(
                "Local price cache contains daily bars; requested timeframe %s will be approximated via daily data.",
                timeframe,
            )
        return local

    fetcher = YahooFinanceDataFetcher()
    interval_map = {
        "1Min": "1m",
        "5Min": "5m",
        "15Min": "15m",
        "30Min": "30m",
        "1Hour": "60m",
        "1Day": "1d",
    }
    interval = interval_map.get(timeframe, "1d")
    lookback_days = max(5, int((end - start).days) + 5)
    range_ = f"{lookback_days}d" if lookback_days <= 720 else "max"
    frame = fetcher.fetch_price_history(ticker, range_=range_, interval=interval)
    if frame.empty:
        _LOG.warning("Yahoo price data request returned empty frame for %s", ticker)
        local = _slice_price_frame(_load_local_price_series(ticker), start, end, ticker=ticker)
        if not local.empty:
            return local
        raise RuntimeError(f"No price data available for {ticker} via Yahoo or local cache")
    frame.index = pd.to_datetime(frame.index).tz_convert("UTC").tz_localize(None)
    filtered = frame[(frame.index >= start) & (frame.index <= end)].copy()
    if filtered.empty:
        _LOG.warning("Filtered Yahoo data empty for %s between %s and %s", ticker, start, end)
        local = _slice_price_frame(_load_local_price_series(ticker), start, end, ticker=ticker)
        if not local.empty:
            return local
        raise RuntimeError(f"No price data available for {ticker} in requested window")
    return filtered

def fetch_gdelt_articles(
    ticker: str,
    start: datetime,
    end: datetime,
    client: Optional[GDELTClient] = None,
) -> List[Dict]:
    if _OFFLINE_MODE:
        return []

    client = client or GDELTClient()
    articles: List[Dict] = []
    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + timedelta(hours=6), end)
        params = {
            "query": ticker,
            "mode": "ArtList",
            "format": "JSON",
            "maxrecords": "250",
            "startdatetime": chunk_start.strftime("%Y%m%d%H%M%S"),
            "enddatetime": chunk_end.strftime("%Y%m%d%H%M%S"),
            "sort": "DateAsc",
        }
        try:
            payload = client._request(params)
        except Exception as exc:  # pragma: no cover - network guard
            _LOG.warning("GDELT article fetch failed for %s: %s", ticker, exc)
            break
        new_articles = payload.get("articles", [])
        if not new_articles:
            _LOG.debug("No GDELT articles for %s between %s and %s", ticker, chunk_start, chunk_end)
        articles.extend(new_articles)
        chunk_start = chunk_end
    return articles


def fetch_gdelt_timeline(
    ticker: str,
    start: datetime,
    end: datetime,
    minutes: int = 60,
    client: Optional[GDELTClient] = None,
) -> List[Tuple[datetime, float]]:
    if _OFFLINE_MODE:
        return []

    client = client or GDELTClient()
    try:
        payload = client._request(
            {
                "query": ticker,
                "mode": "TimelineTone",
                "format": "JSON",
                "startdatetime": start.strftime("%Y%m%d%H%M%S"),
                "enddatetime": end.strftime("%Y%m%d%H%M%S"),
                "timelineminutes": str(minutes),
            }
        )
    except Exception as exc:  # pragma: no cover - network guard
        _LOG.warning("GDELT timeline fetch failed for %s: %s", ticker, exc)
        return []
    series_list = payload.get("timeline", [])
    points: List[Tuple[datetime, float]] = []
    for series in series_list:
        data = series.get("data", [])
        for point in data:
            date_str = point.get("date") or point.get("datetime")
            if not date_str:
                continue
            try:
                timestamp = datetime.strptime(date_str, "%Y%m%dT%H%M%SZ")
            except ValueError:
                continue
            tone = float(point.get("value", point.get("tone", 0.0)))
            points.append((timestamp, tone))
    points.sort(key=lambda pair: pair[0])
    return points
    return articles


__all__ = ["fetch_price_bars", "fetch_gdelt_articles"]
