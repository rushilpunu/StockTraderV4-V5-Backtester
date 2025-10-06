"""Yahoo Finance data service with rate limiting and calendar validation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Iterable, List, Optional, Sequence

import logging

from Traderv5.calendars import TradingCalendar
from Traderv5.configuration import CalendarSettings, DataSettings
from Traderv5.http_client import RateLimitedHttpClient

_LOG = logging.getLogger("traderv5.yahoo")


def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _date_range_span(start: date, end: date) -> tuple[int, int]:
    start_ts = int(_ensure_utc(datetime.combine(start, time.min)).timestamp())
    end_inclusive = datetime.combine(end, time.max)
    end_ts = int(_ensure_utc(end_inclusive).timestamp())
    return start_ts, end_ts


@dataclass(frozen=True)
class PriceBar:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class YahooFinanceService:
    BASE_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"

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

    def fetch_daily_bars(
        self,
        symbol: str,
        start: date,
        end: date,
        *,
        include_prepost: bool = False,
    ) -> List[PriceBar]:
        if start > end:
            raise ValueError("start must be on or before end")
        trading_days = self.calendar.trading_days(start, end)
        if not trading_days:
            return []
        period1, period2 = _date_range_span(start, end + timedelta(days=1))
        params = {
            "interval": "1d",
            "period1": str(period1),
            "period2": str(period2),
            "includePrePost": "true" if include_prepost else "false",
            "events": "div,splits",
        }
        url = self.BASE_URL.format(symbol=symbol)
        payload = self.http_client.get_json(url, params=params)
        chart = (payload.get("chart") or {}).get("result")
        if not chart:
            raise RuntimeError(f"Yahoo Finance returned no chart data for {symbol}")
        result = chart[0]
        timestamps = result.get("timestamp") or []
        indicators = result.get("indicators", {})
        quote = (indicators.get("quote") or [{}])[0]
        opens = quote.get("open") or []
        highs = quote.get("high") or []
        lows = quote.get("low") or []
        closes = quote.get("close") or []
        volumes = quote.get("volume") or []

        bars: List[PriceBar] = []
        for ts, o, h, l, c, v in zip(timestamps, opens, highs, lows, closes, volumes):
            if any(value is None for value in (ts, o, h, l, c, v)):
                continue
            timestamp = datetime.fromtimestamp(int(ts), tz=timezone.utc)
            bars.append(
                PriceBar(
                    symbol=symbol.upper(),
                    timestamp=timestamp,
                    open=float(o),
                    high=float(h),
                    low=float(l),
                    close=float(c),
                    volume=float(v),
                )
            )

        missing = self._validate_completeness(trading_days, bars)
        if missing:
            raise ValueError(
                f"Missing Yahoo price bars for {symbol}: {', '.join(sorted(str(day) for day in missing))}"
            )
        return bars

    def _validate_completeness(self, trading_days: Sequence[date], bars: Sequence[PriceBar]) -> List[date]:
        expected = set(trading_days)
        observed = {bar.timestamp.date() for bar in bars}
        missing = sorted(expected - observed)
        if missing:
            _LOG.warning("Price data missing %s trading days", len(missing))
        return missing


__all__ = ["YahooFinanceService", "PriceBar"]
