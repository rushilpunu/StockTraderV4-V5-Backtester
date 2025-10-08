"""Market data and sentiment loaders for backtests."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from functools import lru_cache
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

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_DATA_DIR = _REPO_ROOT / "data"
_DEFAULT_MULTI_CSV = _DEFAULT_DATA_DIR / "stocks.csv"

def _parse_env_paths(key: str) -> List[Path]:
    raw = os.getenv(key, "").strip()
    if not raw:
        return []
    paths: List[Path] = []
    for entry in raw.split(os.pathsep):
        entry = entry.strip()
        if not entry:
            continue
        paths.append(Path(entry).expanduser().resolve())
    return paths


_LOCAL_CSV_FILES: List[Path] = _parse_env_paths("BACKTEST_PRICE_CSV")
if _DEFAULT_MULTI_CSV.exists():
    # Ensure the repository-shipped dataset is considered last so env overrides win.
    _LOCAL_CSV_FILES.append(_DEFAULT_MULTI_CSV.resolve())


@lru_cache(maxsize=8)
def _load_multi_csv(path: Path) -> Dict[str, pd.DataFrame]:
    if not path.exists() or not path.is_file():
        return {}
    try:
        frame = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - defensive guard for malformed data
        _LOG.warning("Failed to read local CSV data at %s: %s", path, exc)
        return {}
    if frame.empty:
        return {}

    column_map = {col.lower(): col for col in frame.columns}
    symbol_key = column_map.get("symbol") or column_map.get("ticker")
    date_key = (
        column_map.get("date")
        or column_map.get("timestamp")
        or column_map.get("datetime")
    )
    if symbol_key is None or date_key is None:
        _LOG.warning(
            "Local CSV %s missing symbol/date columns; available columns=%s",
            path,
            list(frame.columns),
        )
        return {}

    try:
        frame[date_key] = pd.to_datetime(frame[date_key], errors="coerce")
    except Exception as exc:
        _LOG.warning("Failed to parse dates for %s: %s", path, exc)
        return {}
    frame = frame.dropna(subset=[date_key])
    if frame.empty:
        return {}

    ohlc_candidates = {
        "open": ["open", "opn"],
        "high": ["high", "hi"],
        "low": ["low", "lo"],
        "close": ["close", "adjclose", "adj_close", "price"],
    }
    volume_candidates = ["volume", "vol"]

    grouped: Dict[str, pd.DataFrame] = {}
    for symbol, group in frame.groupby(symbol_key):
        cleaned = group.sort_values(date_key).set_index(date_key)
        cleaned.index = pd.to_datetime(cleaned.index).tz_localize(None)
        base = pd.DataFrame(index=cleaned.index)
        close_series = None

        for field, candidates in ohlc_candidates.items():
            series = None
            for candidate in candidates:
                column = column_map.get(candidate)
                if column and column in cleaned:
                    series = cleaned[column].astype(float)
                    break
            if series is None:
                if field == "close" and close_series is None:
                    continue
                continue
            base[field] = series
            if field == "close":
                close_series = series

        if close_series is None:
            # Fall back to a generic price column if available.
            price_col = column_map.get("price")
            if price_col and price_col in cleaned:
                close_series = cleaned[price_col].astype(float)
                base["close"] = close_series
        if close_series is None or close_series.empty:
            continue

        for field in ("open", "high", "low"):
            if field not in base or base[field].isna().all():
                base[field] = close_series

        volume_series = None
        for candidate in volume_candidates:
            column = column_map.get(candidate)
            if column and column in cleaned:
                try:
                    volume_series = cleaned[column].astype(float)
                except Exception:
                    volume_series = cleaned[column].apply(pd.to_numeric, errors="coerce")
                break
        if volume_series is None:
            volume_series = pd.Series(0.0, index=base.index)
        base["volume"] = volume_series.fillna(0.0)

        base = base[["open", "high", "low", "close", "volume"]]
        base = base.astype(float)
        grouped[str(symbol).upper()] = base

    return grouped


def _load_local_price_series(ticker: str) -> Optional[pd.DataFrame]:
    symbol = ticker.upper()
    for path in _LOCAL_CSV_FILES:
        cache = _load_multi_csv(path)
        frame = cache.get(symbol)
        if frame is not None and not frame.empty:
            return frame
    return None


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

    local_frame = _load_local_price_series(ticker)
    if local_frame is not None:
        filtered = local_frame.copy()
        filtered = filtered[(filtered.index >= start) & (filtered.index <= end)]
        if not filtered.empty:
            _LOG.info("Loaded %s price bars for %s from local dataset", len(filtered), ticker)
            return filtered

    if _OFFLINE_MODE:
        return _synthetic_price_series(ticker, start, end, timeframe)

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
        local_frame = _load_local_price_series(ticker)
        if local_frame is not None:
            filtered = local_frame[(local_frame.index >= start) & (local_frame.index <= end)]
            if not filtered.empty:
                _LOG.info("Loaded %s price bars for %s from local dataset", len(filtered), ticker)
                return filtered
        return _synthetic_price_series(ticker, start, end, timeframe)
    frame.index = pd.to_datetime(frame.index).tz_convert("UTC").tz_localize(None)
    filtered = frame[(frame.index >= start) & (frame.index <= end)].copy()
    if filtered.empty:
        _LOG.warning("Filtered Yahoo data empty for %s between %s and %s", ticker, start, end)
        local_frame = _load_local_price_series(ticker)
        if local_frame is not None:
            filtered = local_frame[(local_frame.index >= start) & (local_frame.index <= end)]
            if not filtered.empty:
                _LOG.info("Loaded %s price bars for %s from local dataset", len(filtered), ticker)
                return filtered
        return _synthetic_price_series(ticker, start, end, timeframe)
    return filtered


def _synthetic_price_series(ticker: str, start: datetime, end: datetime, timeframe: str) -> pd.DataFrame:
    """Fallback generator that approximates a realistic price random walk."""

    freq_map = {
        "1Min": "1min",
        "5Min": "5min",
        "15Min": "15min",
        "30Min": "30min",
        "1Hour": "1H",
        "1Day": "1D",
    }
    freq = freq_map.get(timeframe, "1D")
    index = pd.date_range(start=start, end=end, freq=freq)
    if index.empty:
        index = pd.date_range(start=start, periods=120, freq=freq)

    seed = abs(hash((ticker.upper(), freq))) % (2**32)
    rng = np.random.default_rng(seed)
    steps = len(index)
    if steps < 2:
        base_price = 35.0 + (seed % 120) * 0.5
        return pd.DataFrame(
            {
                "open": [base_price],
                "high": [base_price],
                "low": [base_price],
                "close": [base_price],
                "volume": [250_000],
            },
            index=index,
        )

    # Approximate the length of each step in trading minutes to scale drift/volatility.
    step_minutes = max(1.0, (index[1] - index[0]).total_seconds() / 60.0)
    trading_minutes_per_day = 390.0
    minutes_per_year = trading_minutes_per_day * 252.0

    annual_drift = 0.08  # ~8% annualised drift for equities.
    annual_vol = 0.35    # ~35% annualised volatility.

    drift_per_minute = np.log1p(annual_drift) / minutes_per_year
    vol_per_minute = annual_vol / np.sqrt(minutes_per_year)

    drift = drift_per_minute * step_minutes
    volatility = vol_per_minute * np.sqrt(step_minutes)

    base_price = 35.0 + (seed % 120) * 0.5
    log_price = np.log(base_price)
    log_path = [log_price]
    mean_reversion = 0.12  # pulls the path back towards long-run drift.

    for _ in range(1, steps):
        shock = rng.normal(drift, volatility)
        deviation = log_path[-1] - (log_price + drift * len(log_path))
        shock -= mean_reversion * deviation * 0.01
        log_path.append(log_path[-1] + shock)

    prices_arr = np.exp(log_path)
    prices_arr = np.clip(prices_arr, 1.0, None)

    highs = prices_arr * (1.0 + rng.normal(0.002, 0.01, size=steps))
    lows = prices_arr * (1.0 - rng.normal(0.002, 0.01, size=steps))
    opens = prices_arr * (1.0 + rng.normal(0.0, 0.0025, size=steps))
    volumes = np.maximum(50_000, rng.normal(250_000, 60_000, size=steps))

    frame = pd.DataFrame(
        {
            "open": opens,
            "high": np.maximum.reduce([highs, opens, prices_arr]),
            "low": np.minimum.reduce([lows, opens, prices_arr]),
            "close": prices_arr,
            "volume": volumes,
        },
        index=index,
    )
    _LOG.info(
        "Generated synthetic price series for %s (%d bars, drift=%.6f, vol=%.6f)",
        ticker,
        steps,
        drift,
        volatility,
    )
    return frame


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
