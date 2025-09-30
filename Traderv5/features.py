"""Feature engineering for Trader V5."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from Traderv5.data_sources import GDELTWindow


def _to_naive(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if index.tz is None:
        return index
    return index.tz_convert("UTC").tz_localize(None)


def _infer_bar_minutes(index: pd.DatetimeIndex) -> int:
    if len(index) < 2:
        return 60
    diffs = index.to_series().diff().dropna()
    if diffs.empty:
        return 60
    median_delta = diffs.median()
    minutes = max(1, int(median_delta.total_seconds() // 60))
    return minutes


def _window_counts(article_times: np.ndarray, reference_times: np.ndarray, window: float) -> np.ndarray:
    """Return counts of articles within `window` minutes preceding each reference time."""
    counts = np.zeros(len(reference_times), dtype=float)
    if len(article_times) == 0:
        return counts
    left = 0
    right = 0
    window_seconds = window * 60.0
    for idx, ref in enumerate(reference_times):
        # advance left bound
        while left < len(article_times) and (ref - article_times[left]) > window_seconds:
            left += 1
        right = max(right, left)
        while right < len(article_times) and (article_times[right] - ref) <= 0:
            right += 1
        counts[idx] = max(0, right - left)
    return counts


def _window_average(article_times: np.ndarray, article_values: np.ndarray, reference_times: np.ndarray, window: float) -> np.ndarray:
    averages = np.zeros(len(reference_times), dtype=float)
    if len(article_times) == 0:
        return averages
    left = 0
    right = 0
    window_seconds = window * 60.0
    for idx, ref in enumerate(reference_times):
        while left < len(article_times) and (ref - article_times[left]) > window_seconds:
            left += 1
        right = max(right, left)
        while right < len(article_times) and (article_times[right] - ref) <= 0:
            right += 1
        if right > left:
            averages[idx] = float(article_values[left:right].mean())
        else:
            averages[idx] = 0.0
    return averages


def _as_epoch_seconds(index: pd.DatetimeIndex) -> np.ndarray:
    return index.view(np.int64) // 10**9


@dataclass
class FeatureEngineerConfig:
    sentiment_windows: Sequence[int] = (15, 60, 240)
    article_windows: Sequence[int] = (60, 180, 720)
    price_return_windows: Sequence[int] = (1, 3, 6, 24)
    momentum_window: int = 12
    label_horizon_minutes: int = 90
    positive_threshold: float = 0.0035
    negative_threshold: float = -0.0035


class FeatureEngineer:
    def __init__(self, config: FeatureEngineerConfig | None = None) -> None:
        self.config = config or FeatureEngineerConfig()

    def build_training_frame(
        self,
        ticker: str,
        price_bars: pd.DataFrame,
        gdelt_window: GDELTWindow,
    ) -> pd.DataFrame:
        if price_bars.empty:
            return pd.DataFrame()
        df = price_bars.copy()
        df.index = _to_naive(df.index)
        df = df.sort_index()

        bar_minutes = _infer_bar_minutes(df.index)
        if "close" not in df.columns:
            raise ValueError("price_bars requires 'close' column")

        df["close"] = df["close"].astype(float)
        df["volume"] = df.get("volume", pd.Series(index=df.index, dtype=float)).fillna(0.0).astype(float)

        df["return_1"] = df["close"].pct_change().fillna(0.0)
        for window in self.config.price_return_windows:
            df[f"return_{window}"] = df["close"].pct_change(window).fillna(0.0)
        df["volatility_6"] = df["return_1"].rolling(6, min_periods=2).std().fillna(0.0)
        df["volume_z"] = (df["volume"] - df["volume"].rolling(24, min_periods=4).mean()) / (
            df["volume"].rolling(24, min_periods=4).std().replace({0: np.nan})
        )
        df["volume_z"] = df["volume_z"].replace({np.nan: 0.0, np.inf: 0.0, -np.inf: 0.0})

        tone_series = self._build_tone_series(gdelt_window)
        article_features = self._build_article_features(gdelt_window, df.index)

        for win in self.config.sentiment_windows:
            tone_window = tone_series.rolling(f"{win}min").mean().reindex(df.index, method="ffill").fillna(0.0)
            df[f"tone_{win}"] = tone_window
            lag_steps = max(1, int(win / max(bar_minutes, 1)))
            df[f"tone_delta_{win}"] = tone_window - tone_window.shift(lag_steps).fillna(0.0)

        for name, values in article_features.items():
            df[name] = values

        # Labeling
        horizon_steps = max(1, int(self.config.label_horizon_minutes / bar_minutes))
        future_close = df["close"].shift(-horizon_steps)
        df["forward_return"] = (future_close - df["close"]) / df["close"]
        df["label"] = 0
        df.loc[df["forward_return"] >= self.config.positive_threshold, "label"] = 1
        df.loc[df["forward_return"] <= self.config.negative_threshold, "label"] = -1

        df["ticker"] = ticker
        return df.dropna()

    def _build_tone_series(self, gdelt_window: GDELTWindow) -> pd.Series:
        if not gdelt_window.timeline:
            return pd.Series(dtype=float)
        timeline_df = pd.DataFrame(
            {
                "timestamp": [ts for ts, _ in gdelt_window.timeline],
                "tone": [tone for _, tone in gdelt_window.timeline],
            }
        )
        timeline_df["timestamp"] = pd.to_datetime(timeline_df["timestamp"], utc=True).dt.tz_convert(None)
        timeline_df = timeline_df.drop_duplicates(subset="timestamp").set_index("timestamp").sort_index()
        return timeline_df["tone"]

    def _build_article_features(
        self, gdelt_window: GDELTWindow, reference_index: pd.DatetimeIndex
    ) -> Dict[str, pd.Series]:
        if not gdelt_window.articles:
            zero = pd.Series(np.zeros(len(reference_index)), index=reference_index)
            return {
                "article_count_60": zero,
                "article_count_180": zero,
                "article_avg_tone_60": zero,
            }
        timestamps: List[datetime] = []
        tones: List[float] = []
        for entry in gdelt_window.articles:
            raw_timestamp = entry.get("seendate") or entry.get("publishtime") or entry.get("date")
            if not raw_timestamp:
                continue
            ts = self._parse_timestamp(raw_timestamp)
            if ts is None:
                continue
            timestamps.append(ts)
            try:
                tones.append(float(entry.get("tone", 0.0)))
            except (TypeError, ValueError):
                tones.append(0.0)
        if not timestamps:
            zero = pd.Series(np.zeros(len(reference_index)), index=reference_index)
            return {
                "article_count_60": zero,
                "article_count_180": zero,
                "article_avg_tone_60": zero,
            }
        article_index = pd.to_datetime(timestamps, utc=True).tz_convert(None)
        order = np.argsort(article_index.values)
        article_index = article_index.take(order)
        tones_array = np.asarray(tones, dtype=float)[order]
        reference_index = reference_index.tz_localize(None) if reference_index.tz is not None else reference_index

        article_seconds = _as_epoch_seconds(article_index)
        reference_seconds = _as_epoch_seconds(reference_index)

        counts_60 = _window_counts(article_seconds, reference_seconds, 60)
        counts_180 = _window_counts(article_seconds, reference_seconds, 180)
        avg_tone_60 = _window_average(article_seconds, tones_array, reference_seconds, 60)

        return {
            "article_count_60": pd.Series(counts_60, index=reference_index),
            "article_count_180": pd.Series(counts_180, index=reference_index),
            "article_avg_tone_60": pd.Series(avg_tone_60, index=reference_index),
        }

    def build_live_features(
        self,
        price_context: pd.DataFrame,
        gdelt_window: GDELTWindow,
    ) -> Optional[pd.Series]:
        frame = self.build_training_frame("live", price_context, gdelt_window)
        if frame.empty:
            return None
        latest = frame.iloc[-1]
        feature_columns = self.feature_columns(frame)
        return latest[feature_columns]

    def feature_columns(self, frame: Optional[pd.DataFrame] = None) -> List[str]:
        if frame is not None:
            return [col for col in frame.columns if col not in {"label", "forward_return", "ticker"}]
        # fallback ordering
        columns: List[str] = [
            "close",
            "volume",
            "return_1",
            *[f"return_{window}" for window in self.config.price_return_windows],
            "volatility_6",
            "volume_z",
        ]
        for win in self.config.sentiment_windows:
            columns.append(f"tone_{win}")
            columns.append(f"tone_delta_{win}")
        columns.extend([
            "article_count_60",
            "article_count_180",
            "article_avg_tone_60",
        ])
        return columns

    @staticmethod
    def _parse_timestamp(raw: str) -> Optional[datetime]:
        formats = [
            "%Y%m%dT%H%M%SZ",
            "%Y%m%dT%H%M%S",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%d %H:%M:%S",
        ]
        for fmt in formats:
            try:
                return datetime.strptime(raw, fmt).replace(tzinfo=None)
            except ValueError:
                continue
        if raw.endswith("Z"):
            try:
                return datetime.strptime(raw[:-1], "%Y%m%dT%H%M%S")
            except ValueError:
                return None
        return None


__all__ = ["FeatureEngineer", "FeatureEngineerConfig"]
