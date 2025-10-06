"""GDELT sentiment service with rate limiting, caching, and daily summaries."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import logging
import statistics
import time

from Traderv5.calendars import TradingCalendar
from Traderv5.configuration import CalendarSettings, DataSettings
from Traderv5.http_client import RateLimitedHttpClient

_LOG = logging.getLogger("traderv5.gdelt")

_GDELT_BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
_DATETIME_FORMAT = "%Y%m%d%H%M%S"


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _format_timestamp(value: datetime) -> str:
    return _as_utc(value).strftime(_DATETIME_FORMAT)


def _parse_timestamp(raw: str) -> Optional[datetime]:
    if not raw:
        return None
    formats = [
        "%Y%m%dT%H%M%SZ",
        "%Y%m%d%H%M%S",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d %H:%M:%S",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(raw, fmt)
        except ValueError:
            continue
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    if raw.endswith("Z"):
        return _parse_timestamp(raw[:-1])
    return None


@dataclass(frozen=True)
class GDELTArticle:
    symbol: str
    published_at: datetime
    tone: float
    title: str
    url: str
    source: str
    raw: Dict[str, object]


@dataclass(frozen=True)
class GDELTSentimentSummary:
    symbol: str
    date: date
    average_tone: float
    tone_std: float
    article_count: int
    positive_article_count: int
    negative_article_count: int
    timeline_points: int


@dataclass(frozen=True)
class GDELTWindow:
    symbol: str
    start: datetime
    end: datetime
    summaries: List[GDELTSentimentSummary]
    articles: List[GDELTArticle]
    timeline: List[Tuple[datetime, float]]


class GDELTService:
    def __init__(
        self,
        http_client: RateLimitedHttpClient,
        *,
        calendar: Optional[TradingCalendar] = None,
        calendar_settings: Optional[CalendarSettings] = None,
        data_settings: Optional[DataSettings] = None,
    ) -> None:
        self.http_client = http_client
        self.calendar = calendar or TradingCalendar(calendar_settings or CalendarSettings())
        self.data_settings = data_settings

    def fetch_sentiment_window(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        *,
        query_terms: Optional[List[str]] = None,
    ) -> GDELTWindow:
        if start >= end:
            raise ValueError("start must be before end")
        start = _as_utc(start)
        end = _as_utc(end)
        query = self._build_query(symbol, query_terms)
        timeline = self._fetch_timeline(query, start, end)
        articles = self._fetch_articles(query, start, end)
        summaries = self._summaries_for_days(symbol, start.date(), end.date(), timeline, articles)
        return GDELTWindow(
            symbol=symbol.upper(),
            start=start,
            end=end,
            summaries=summaries,
            articles=articles,
            timeline=timeline,
        )

    def _fetch_timeline(self, query: str, start: datetime, end: datetime) -> List[Tuple[datetime, float]]:
        _LOG.info(
            "GDELT timeline: query=%s, start=%s, end=%s, minutes_per_point=%s",
            query,
            _format_timestamp(start),
            _format_timestamp(end),
            int(self._gdelt_chunk_minutes()),
        )
        params = {
            "query": query,
            "mode": "TimelineTone",
            "format": "JSON",
            "startdatetime": _format_timestamp(start),
            "enddatetime": _format_timestamp(end),
            "timelineminutes": str(int(self._gdelt_chunk_minutes())),
        }
        payload = self.http_client.get_json(_GDELT_BASE_URL, params=params)
        series = payload.get("timeline", []) if isinstance(payload, dict) else []
        points: List[Tuple[datetime, float]] = []
        for block in series:
            for entry in block.get("data", []):
                raw_date = entry.get("date") or entry.get("datetime")
                tone_raw = entry.get("tone") or entry.get("value")
                if raw_date is None or tone_raw is None:
                    continue
                timestamp = _parse_timestamp(str(raw_date))
                if timestamp is None:
                    continue
                try:
                    tone = float(tone_raw)
                except (TypeError, ValueError):
                    continue
                points.append((_as_utc(timestamp), tone))
        points.sort(key=lambda item: item[0])
        _LOG.info("GDELT timeline: %d points collected", len(points))
        return points

    def _fetch_articles(self, query: str, start: datetime, end: datetime) -> List[GDELTArticle]:
        chunk_minutes = max(int(self._gdelt_chunk_minutes()), 60)
        articles: List[GDELTArticle] = []
        window_start = start
        chunk_idx = 0
        _LOG.info(
            "GDELT articles: query=%s, start=%s, end=%s, chunk_minutes=%s",
            query,
            _format_timestamp(start),
            _format_timestamp(end),
            chunk_minutes,
        )
        while window_start < end:
            window_end = min(window_start + timedelta(minutes=chunk_minutes), end)
            chunk_idx += 1
            _LOG.info(
                "GDELT articles: chunk %d | %s → %s",
                chunk_idx,
                _format_timestamp(window_start),
                _format_timestamp(window_end),
            )
            params = {
                "query": query,
                "mode": "ArtList",
                "format": "JSON",
                "maxrecords": "250",
                "sort": "DateAsc",
                "startdatetime": _format_timestamp(window_start),
                "enddatetime": _format_timestamp(window_end),
            }
            payload = self.http_client.get_json(_GDELT_BASE_URL, params=params)
            chunk_count = 0
            for entry in payload.get("articles", []) if isinstance(payload, dict) else []:
                published_raw = entry.get("seendate") or entry.get("publishdate") or entry.get("date")
                timestamp = _parse_timestamp(str(published_raw))
                if timestamp is None:
                    continue
                try:
                    tone = float(entry.get("tone", 0.0))
                except (TypeError, ValueError):
                    tone = 0.0
                article = GDELTArticle(
                    symbol=symbol.upper() if (symbol := query) else query,
                    published_at=_as_utc(timestamp),
                    tone=tone,
                    title=str(entry.get("title", "")),
                    url=str(entry.get("sourceurl", "")),
                    source=str(entry.get("source", "gdelt")),
                    raw=entry,
                )
                articles.append(article)
                chunk_count += 1
            _LOG.info(
                "GDELT articles: chunk %d fetched %d articles (cumulative %d)",
                chunk_idx,
                chunk_count,
                len(articles),
            )
            window_start = window_end
            pause = self._gdelt_pause_seconds()
            if pause > 0:
                _LOG.info("GDELT articles: pausing %.2fs to respect rate limits", pause)
                time.sleep(pause)
        articles.sort(key=lambda article: article.published_at)
        _LOG.info("GDELT articles: total %d articles collected", len(articles))
        return articles

    @staticmethod
    def _build_query(symbol: str, terms: Optional[List[str]]) -> str:
        tokens: List[str] = []
        base = str(symbol).strip()
        if base:
            tokens.append(base)
        if terms:
            for term in terms:
                t = str(term).strip()
                if not t:
                    continue
                if " " in t:
                    tokens.append(f'"{t}"')
                else:
                    tokens.append(t)
        if not tokens:
            return base
        return f"({' OR '.join(tokens)})"

    def _summaries_for_days(
        self,
        symbol: str,
        start_day: date,
        end_day: date,
        timeline: Sequence[Tuple[datetime, float]],
        articles: Sequence[GDELTArticle],
    ) -> List[GDELTSentimentSummary]:
        trading_days = self.calendar.trading_days(start_day, end_day)
        timeline_by_day: Dict[date, List[float]] = {day: [] for day in trading_days}
        for timestamp, tone in timeline:
            day = timestamp.date()
            if day in timeline_by_day:
                timeline_by_day[day].append(tone)
        article_by_day: Dict[date, List[GDELTArticle]] = {day: [] for day in trading_days}
        for article in articles:
            day = article.published_at.date()
            if day in article_by_day:
                article_by_day[day].append(article)
        summaries: List[GDELTSentimentSummary] = []
        for day in trading_days:
            timeline_tones = timeline_by_day.get(day, [])
            article_group = article_by_day.get(day, [])
            article_tones = [article.tone for article in article_group]
            combined: List[float] = []
            if article_tones:
                combined.extend(article_tones)
            if timeline_tones:
                combined.extend(timeline_tones)
            if combined:
                average = float(sum(combined) / len(combined))
                tone_std = float(statistics.pstdev(combined)) if len(combined) > 1 else 0.0
            else:
                average = 0.0
                tone_std = 0.0
            positive = sum(1 for tone in article_tones if tone > 0)
            negative = sum(1 for tone in article_tones if tone < 0)
            summaries.append(
                GDELTSentimentSummary(
                    symbol=symbol.upper(),
                    date=day,
                    average_tone=average,
                    tone_std=tone_std,
                    article_count=len(article_group),
                    positive_article_count=positive,
                    negative_article_count=negative,
                    timeline_points=len(timeline_tones),
                )
            )
        return summaries

    def _gdelt_chunk_minutes(self) -> float:
        if self.data_settings is not None:
            return float(self.data_settings.gdelt_chunk_minutes)
        return 240.0

    def _gdelt_pause_seconds(self) -> float:
        if self.data_settings is not None:
            return float(self.data_settings.gdelt_pause_seconds)
        return 0.0


__all__ = [
    "GDELTService",
    "GDELTArticle",
    "GDELTSentimentSummary",
    "GDELTWindow",
]
