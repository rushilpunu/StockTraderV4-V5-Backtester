"""Trading calendar and settlement utilities for Trader V5."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import List, Optional, Set
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    GoodFriday,
    USFederalHolidayCalendar,
)
try:  # pragma: no cover - optional dependency
    import pandas_market_calendars as mcal
except Exception:  # pragma: no cover - fallback when unavailable
    mcal = None

from Traderv5.configuration import CalendarSettings


def _to_date(value: date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    raise TypeError(f"Expected date or datetime, got {type(value)!r}")


def _normalize_range(start: date | datetime, end: date | datetime) -> tuple[date, date]:
    start_date = _to_date(start)
    end_date = _to_date(end)
    if start_date > end_date:
        start_date, end_date = end_date, start_date
    return start_date, end_date


class _XNYSHolidayCalendar(AbstractHolidayCalendar):
    rules = USFederalHolidayCalendar.rules + [GoodFriday]


def _federal_holidays(start: date, end: date) -> Set[date]:
    cal = USFederalHolidayCalendar()
    holidays = set(dt.date() for dt in cal.holidays(start=start, end=end).to_pydatetime())
    good_fridays = set(dt.date() for dt in GoodFriday.dates(start=start, end=end).to_pydatetime())
    holidays.update(good_fridays)
    return holidays


@dataclass
class TradingCalendar:
    settings: CalendarSettings
    # Fast mode avoids using pandas_market_calendars even if available to
    # eliminate any potential slowdowns or hangs in schedule generation.
    # The simple weekday/holiday generator below is deterministic and fast.
    fast_mode: bool = True

    def trading_days(self, start: date | datetime, end: date | datetime) -> List[date]:
        start_date, end_date = _normalize_range(start, end)
        # Hard safety bound to prevent excessive ranges
        if (end_date - start_date).days > 366 * 10:
            raise ValueError("Requested trading day range is too large")
        # Prefer fast path unless explicitly disabled
        if not self.fast_mode and mcal is not None:
            cal_name = self.settings.holiday_calendar or self.settings.exchange
            try:
                calendar = mcal.get_calendar(cal_name)
                schedule = calendar.schedule(start_date=start_date, end_date=end_date)
                return [ts.date() for ts in schedule.index.to_pydatetime()]
            except Exception:
                # Fall back to simple generator on any issue
                pass
        holidays = self._holiday_set(start_date, end_date)
        days: List[date] = []
        current = start_date
        # Safety counter to avoid unexpected infinite loops
        max_steps = (end_date - start_date).days + 10
        steps = 0
        while current <= end_date and steps <= max_steps:
            if self._is_weekday(current) and current not in holidays:
                days.append(current)
            current += timedelta(days=1)
            steps += 1
        return days

    def is_trading_day(self, target: date | datetime) -> bool:
        day = _to_date(target)
        if not self._is_weekday(day):
            return False
        holidays = self._holiday_set(day - timedelta(days=10), day + timedelta(days=10))
        return day not in holidays

    def next_trading_day(self, target: date | datetime, *, include_target: bool = False) -> date:
        day = _to_date(target)
        if include_target and self.is_trading_day(day):
            return day
        current = day + timedelta(days=1)
        # Safety: cap search to 365 iterations
        for _ in range(366):
            if self.is_trading_day(current):
                return current
            current += timedelta(days=1)
        # Fallback: return next calendar day if something went wrong
        return day + timedelta(days=1)

    def previous_trading_day(self, target: date | datetime, *, include_target: bool = False) -> date:
        day = _to_date(target)
        if include_target and self.is_trading_day(day):
            return day
        current = day - timedelta(days=1)
        # Safety: cap search to 365 iterations
        for _ in range(366):
            if self.is_trading_day(current):
                return current
            current -= timedelta(days=1)
        # Fallback: return previous calendar day if something went wrong
        return day - timedelta(days=1)

    def calculate_settlement_date(
        self,
        trade_date: date | datetime,
        *,
        settlement_lag: Optional[int] = None,
    ) -> date:
        lag = settlement_lag or self.settings.settlement_lag_days
        if lag <= 0:
            raise ValueError("settlement_lag must be positive")
        day = _to_date(trade_date)
        steps = 0
        current = day
        while steps < lag:
            current += timedelta(days=1)
            if self.is_trading_day(current):
                steps += 1
        return current

    def _holiday_set(self, start: date, end: date) -> Set[date]:
        custom = set(self.settings.custom_holidays)
        if self.settings.exchange.upper() == "XNYS":
            holidays = set(dt.date() for dt in _XNYSHolidayCalendar().holidays(start=start, end=end).to_pydatetime())
        else:
            holidays = _federal_holidays(start, end)
        holidays.update(custom)
        return holidays

    @staticmethod
    def _is_weekday(day: date) -> bool:
        return day.weekday() < 5


def trading_days_between(
    start: date | datetime,
    end: date | datetime,
    *,
    settings: Optional[CalendarSettings] = None,
) -> List[date]:
    calendar = TradingCalendar(settings or CalendarSettings())
    return calendar.trading_days(start, end)


def calculate_settlement_date(
    trade_date: date | datetime,
    *,
    settings: Optional[CalendarSettings] = None,
    settlement_lag: Optional[int] = None,
) -> date:
    calendar = TradingCalendar(settings or CalendarSettings())
    return calendar.calculate_settlement_date(trade_date, settlement_lag=settlement_lag)


__all__ = [
    "TradingCalendar",
    "trading_days_between",
    "calculate_settlement_date",
]
