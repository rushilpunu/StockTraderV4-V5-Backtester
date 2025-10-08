"""Data ingestion utilities for Trader V5."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import logging
import os
import time

import pandas as pd
import statistics
import requests

from Traderv4.GDELT import GDELTClient
from Traderv5.services.gdelt import GDELTSentimentSummary
from collections import defaultdict

_OFFLINE_MODE = os.getenv("BACKTEST_OFFLINE", "1").lower() not in {"0", "false", "no"}
_LOCAL_PRICE_FILE = Path(__file__).resolve().parents[1] / "data" / "prices" / "all_stocks_5yr_subset.csv"
_LOCAL_PRICE_CACHE: Dict[str, pd.DataFrame] = {}

_LOG = logging.getLogger(__name__)


@dataclass
class GDELTWindow:
    ticker: str
    start: datetime
    end: datetime
    timeline_minutes: int
    timeline: List[Tuple[datetime, float]]
    articles: List[Dict[str, Any]]
    summaries: List[GDELTSentimentSummary]


class YahooFinanceDataFetcher:
    """Lightweight Yahoo Finance market data pulls via public chart API."""

    BASE_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"

    def __init__(self, *, session: Optional[requests.Session] = None) -> None:
        self.session = session or requests.Session()

    def _request(self, ticker: str, params: Dict[str, Any], *, retries: int = 5, backoff: float = 1.8) -> Optional[Dict[str, Any]]:
        if _OFFLINE_MODE:
            return None
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
            return self._local_price_history(ticker)
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

    def _local_price_history(self, ticker: str) -> pd.DataFrame:
        symbol = ticker.upper()
        cached = _LOCAL_PRICE_CACHE.get(symbol)
        if cached is not None:
            return cached.copy()
        if not _LOCAL_PRICE_FILE.exists():
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        frame = pd.read_csv(_LOCAL_PRICE_FILE)
        if "date" not in frame.columns or "Name" not in frame.columns:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        frame["date"] = pd.to_datetime(frame["date"])
        subset = frame[frame["Name"].str.upper() == symbol].copy()
        if subset.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        subset = subset.sort_values("date").set_index("date")
        subset = subset[["open", "high", "low", "close", "volume"]]
        subset = subset.apply(pd.to_numeric, errors="coerce").ffill().bfill()
        _LOCAL_PRICE_CACHE[symbol] = subset
        return subset.copy()


def _summaries_from_data(
    ticker: str,
    timeline: List[Tuple[datetime, float]],
    articles: List[Dict[str, Any]],
) -> List[GDELTSentimentSummary]:
    tone_by_day: Dict[date, List[float]] = defaultdict(list)
    for ts, tone in timeline:
        tone_by_day[ts.date()].append(float(tone))

    article_by_day: Dict[date, List[Dict[str, Any]]] = defaultdict(list)
    for article in articles:
        seendate = article.get("seendate") or article.get("publishdate")
        parsed_date: Optional[date] = None
        if isinstance(seendate, str):
            for fmt in ("%Y%m%dT%H%M%SZ", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d"):
                try:
                    parsed_date = datetime.strptime(seendate, fmt).date()
                    break
                except ValueError:
                    continue
        elif isinstance(seendate, datetime):
            parsed_date = seendate.date()
        if parsed_date is None:
            continue
        article_by_day[parsed_date].append(article)

    all_days = sorted(set(tone_by_day) | set(article_by_day))
    summaries: List[GDELTSentimentSummary] = []
    for day in all_days:
        tones = tone_by_day.get(day, [])
        tone_avg = statistics.fmean(tones) if tones else 0.0
        tone_std = statistics.pstdev(tones) if len(tones) > 1 else 0.0
        articles_for_day = article_by_day.get(day, [])
        article_count = len(articles_for_day)
        positive_count = 0
        negative_count = 0
        for article in articles_for_day:
            change = article.get("change")
            tone = article.get("tone")
            value = 0.0
            if isinstance(change, (int, float)):
                value = float(change)
            elif isinstance(tone, (int, float)):
                value = float(tone)
            if value > 0:
                positive_count += 1
            elif value < 0:
                negative_count += 1
        summaries.append(
            GDELTSentimentSummary(
                symbol=ticker.upper(),
                date=day,
                average_tone=float(tone_avg),
                tone_std=float(tone_std),
                article_count=article_count,
                positive_article_count=positive_count,
                negative_article_count=negative_count,
                timeline_points=len(tones),
            )
        )
    return summaries


def _price_derived_window(
    ticker: str,
    start: datetime,
    end: datetime,
    timeline_minutes: int,
) -> GDELTWindow:
    fetcher = YahooFinanceDataFetcher()
    price_df = fetcher._local_price_history(ticker)
    if price_df.empty:
        return GDELTWindow(
            ticker=ticker,
            start=start,
            end=end,
            timeline_minutes=timeline_minutes,
            timeline=[],
            articles=[],
            summaries=[],
        )

    start_norm = pd.Timestamp(start).tz_localize(None).normalize()
    end_norm = pd.Timestamp(end).tz_localize(None).normalize()
    window = price_df[(price_df.index >= start_norm) & (price_df.index <= end_norm)].copy()
    if window.empty:
        return GDELTWindow(
            ticker=ticker,
            start=start,
            end=end,
            timeline_minutes=timeline_minutes,
            timeline=[],
            articles=[],
            summaries=[],
        )

    returns = window["close"].pct_change().fillna(0.0)
    timeline = [
        (ts.tz_localize(timezone.utc), float(returns.loc[ts]))
        for ts in returns.index
    ]
    timeline.sort(key=lambda pair: pair[0])

    prev_close = window["close"].shift(1)
    articles: List[Dict[str, Any]] = []
    for ts, row in window.iterrows():
        prev = prev_close.loc[ts]
        change = 0.0
        if pd.notna(prev) and prev != 0:
            change = float((row["close"] - prev) / prev)
        articles.append(
            {
                "seendate": ts.strftime("%Y%m%dT%H%M%SZ"),
                "publishdate": ts.strftime("%Y%m%dT%H%M%SZ"),
                "title": f"{ticker} close {row['close']:.2f} ({change:+.2%})",
                "domain": "price-feed",
                "url": f"local://price/{ticker}/{ts.date()}",
                "change": change,
            }
        )

    summaries = _summaries_from_data(ticker, timeline, articles)

    return GDELTWindow(
        ticker=ticker,
        start=start,
        end=end,
        timeline_minutes=timeline_minutes,
        timeline=timeline,
        articles=articles,
        summaries=summaries,
    )

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
    if _OFFLINE_MODE:
        return _price_derived_window(ticker, start, end, timeline_minutes)

    client = client or GDELTClient(delay=delay)
    def _gdelt_request(params: Dict[str, str], *, attempts: int = 4) -> Dict[str, Any]:
        for attempt in range(1, attempts + 1):
            try:
                return client._request(params)
            except requests.RequestException as exc:
                wait = max(delay, 1.0) * attempt
                if attempt == attempts:
                    _LOG.error("GDELT request failed for %s after %s attempts: %s", ticker, attempts, exc)
                    raise
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

    try:
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
    except requests.RequestException:
        return _price_derived_window(ticker, start, end, timeline_minutes)
    timeline_raw = timeline_payload.get("timeline", [])
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
        try:
            payload = _gdelt_request(params)
        except requests.RequestException:
            derived = _price_derived_window(ticker, start, end, timeline_minutes)
            articles.extend(derived.articles)
            timeline = derived.timeline if not timeline else timeline
            break
        articles.extend(payload.get("articles", []))
        chunk_start = chunk_end
        time.sleep(max(0.0, delay - 0.2))

    summaries = _summaries_from_data(ticker, timeline, articles)

    return GDELTWindow(
        ticker=ticker,
        start=start,
        end=end,
        timeline_minutes=timeline_minutes,
        timeline=timeline,
        articles=articles,
        summaries=summaries,
    )


__all__ = ["YahooFinanceDataFetcher", "collect_gdelt_window", "GDELTWindow"]
