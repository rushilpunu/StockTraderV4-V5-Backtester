"""Market data and sentiment loaders for backtests."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

try:
    from alpaca_trade_api import REST  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("alpaca-trade-api is required for backtesting data fetch") from exc

from Traderv4.GDELT import GDELTClient
from config.credentials import load_alpaca_credentials

_LOG = logging.getLogger(__name__)


def _alpaca_rest() -> REST:
    creds = load_alpaca_credentials()
    return REST(key_id=creds.api_key, secret_key=creds.api_secret, base_url=creds.base_url)


def _format_timestamp(value: datetime) -> str:
    if value.tzinfo is None:
        return value.strftime("%Y-%m-%dT%H:%M:%SZ")
    return value.astimezone().strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch_price_bars(
    ticker: str,
    start: datetime,
    end: datetime,
    timeframe: str = "15Min",
) -> pd.DataFrame:
    """Fetch historical bars from Alpaca market data."""
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
    if df.empty:
        _LOG.warning("No price data returned for %s", ticker)
    df = df.sort_index()
    return df


def fetch_gdelt_articles(
    ticker: str,
    start: datetime,
    end: datetime,
    client: Optional[GDELTClient] = None,
) -> List[Dict]:
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
        payload = client._request(params)
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
    client = client or GDELTClient()
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
