"""Data ingestion utilities for Trader V5."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

import logging
import os
import time

import pandas as pd
import requests

from Traderv4.GDELT import GDELTClient

_LOG = logging.getLogger(__name__)


@dataclass
class GDELTWindow:
    ticker: str
    start: datetime
    end: datetime
    timeline_minutes: int
    timeline: List[Tuple[datetime, float]]
    articles: List[Dict[str, Any]]
    summaries: List[Any]


class YahooFinanceDataFetcher:
    """Lightweight Yahoo Finance market data pulls via public chart API."""

    BASE_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"

    def __init__(self, *, session: Optional[requests.Session] = None) -> None:
        self.session = session or requests.Session()

    def _request(self, ticker: str, params: Dict[str, Any], *, retries: int = 5, backoff: float = 1.8) -> Optional[Dict[str, Any]]:
        url = self.BASE_URL.format(ticker=ticker)
        attempt = 0
        time.sleep(0.4)
        while attempt < retries:
            attempt += 1
            try:
                response = self.session.get(
                    url,
                    params=params,
                    timeout=10,
                    headers={"User-Agent": "Mozilla/5.0 TraderV5"},
                )
                response.raise_for_status()
            except requests.HTTPError as exc:
                status = exc.response.status_code if hasattr(exc, "response") and exc.response else None
                if status == 429 and attempt < retries:
                    sleep_time = backoff * attempt
                    _LOG.warning("Yahoo chart throttled for %s; retrying in %.1fs", ticker, sleep_time)
                    time.sleep(sleep_time)
                    continue
                _LOG.warning("Yahoo chart request failed for %s: %s", ticker, exc)
                return None
            except requests.RequestException as exc:
                if attempt < retries:
                    time.sleep(backoff * attempt)
                    continue
                _LOG.warning("Yahoo chart request failed for %s: %s", ticker, exc)
                return None
            break
        payload = response.json()
        result = (payload.get("chart") or {}).get("result")
        if not result:
            return None
        return result[0]

    def fetch_price_history(
        self,
        ticker: str,
        *,
        range_: str = "1mo",
        interval: str = "1h",
        include_prepost: bool = False,
    ) -> pd.DataFrame:
        """Return a DataFrame of price bars indexed by UTC timestamp."""
        chart = self._request(
            ticker,
            {
                "range": range_,
                "interval": interval,
                "includePrePost": str(include_prepost).lower(),
                "events": "div,splits",
            },
        )
        if not chart:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        timestamp = chart.get("timestamp") or []
        indicators = chart.get("indicators", {})
        quote = (indicators.get("quote") or [{}])[0]
        frame = pd.DataFrame(quote)
        if frame.empty or not timestamp:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        frame.index = [datetime.fromtimestamp(ts, tz=timezone.utc) for ts in timestamp]
        frame = frame.rename(columns={"close": "close", "open": "open", "volume": "volume"})
        frame = frame[[col for col in ["open", "high", "low", "close", "volume"] if col in frame.columns]]
        frame = frame.astype(float)
        return frame.sort_index()

    def fetch_recent_window(
        self,
        ticker: str,
        *,
        lookback: timedelta = timedelta(days=5),
        interval: str = "1h",
    ) -> pd.DataFrame:
        """Convenience wrapper around `fetch_price_history` using a bounded lookback."""
        days = max(1, int(lookback.total_seconds() // 86400))
        range_ = f"{days}d" if days <= 60 else "max"
        frame = self.fetch_price_history(ticker, range_=range_, interval=interval)
        if frame.empty:
            return frame
        cutoff = datetime.utcnow().replace(tzinfo=timezone.utc) - lookback
        return frame[frame.index >= cutoff]


def collect_gdelt_window(
    ticker: str,
    start: datetime,
    end: datetime,
    *,
    timeline_minutes: int = 60,
    client: Optional[GDELTClient] = None,
    delay: float = 0.75,
) -> GDELTWindow:
    """Fetch timeline and article slices for a ticker/time range."""
    if os.getenv("BACKTEST_DISABLE_GDELT", "0").lower() in {"1", "true", "yes"}:
        return GDELTWindow(
            ticker=ticker,
            start=start,
            end=end,
            timeline_minutes=timeline_minutes,
            timeline=[],
            articles=[],
            summaries=[],
        )
    client = client or GDELTClient(delay=delay)
    def _gdelt_request(params: Dict[str, str], *, attempts: int = 4) -> Dict[str, Any]:
        for attempt in range(1, attempts + 1):
            try:
                return client._request(params)
            except requests.RequestException as exc:
                wait = max(delay, 1.0) * attempt
                if attempt == attempts:
                    _LOG.error(
                        "GDELT request failed for %s after %s attempts: %s",
                        ticker,
                        attempts,
                        exc,
                    )
                    return {}
                _LOG.warning(
                    "GDELT request failed for %s (attempt %s/%s): %s; retrying in %.1fs",
                    ticker,
                    attempt,
                    attempts,
                    exc,
                    wait,
                )
                time.sleep(wait)
        return {}

    timeline_payload = _gdelt_request(
        {
            "query": ticker,
            "mode": "TimelineTone",
            "format": "JSON",
            "startdatetime": start.strftime("%Y%m%d%H%M%S"),
            "enddatetime": end.strftime("%Y%m%d%H%M%S"),
            "timelineminutes": str(timeline_minutes),
        }
    )
    timeline_raw = (timeline_payload or {}).get("timeline", [])
    timeline: List[Tuple[datetime, float]] = []
    for series in timeline_raw:
        for point in series.get("data", []):
            date_str = point.get("date") or point.get("datetime")
            if not date_str:
                continue
            try:
                timestamp = datetime.strptime(date_str, "%Y%m%dT%H%M%SZ")
            except ValueError:
                continue
            tone_val = point.get("value", point.get("tone"))
            try:
                tone = float(tone_val)
            except (TypeError, ValueError):
                continue
            timeline.append((timestamp.replace(tzinfo=timezone.utc), tone))
    timeline.sort(key=lambda pair: pair[0])

    articles: List[Dict[str, Any]] = []
    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + timedelta(hours=4), end)
        params = {
            "query": ticker,
            "mode": "ArtList",
            "format": "JSON",
            "maxrecords": "250",
            "startdatetime": chunk_start.strftime("%Y%m%d%H%M%S"),
            "enddatetime": chunk_end.strftime("%Y%m%d%H%M%S"),
            "sort": "DateAsc",
        }
        payload = _gdelt_request(params)
        articles.extend((payload or {}).get("articles", []))
        chunk_start = chunk_end
        time.sleep(max(0.0, delay - 0.2))

    return GDELTWindow(
        ticker=ticker,
        start=start,
        end=end,
        timeline_minutes=timeline_minutes,
        timeline=timeline,
        articles=articles,
        summaries=[],
    )


__all__ = ["YahooFinanceDataFetcher", "collect_gdelt_window", "GDELTWindow"]
