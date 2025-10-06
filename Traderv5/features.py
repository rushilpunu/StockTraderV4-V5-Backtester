"""Feature engineering for Trader V5."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from Traderv5.services import GDELTSentimentSummary, GDELTWindow, PriceBar


def _infer_bar_minutes(index: pd.DatetimeIndex) -> int:
    if len(index) < 2:
        return 60
    diffs = index.to_series().diff().dropna()
    if diffs.empty:
        return 60
    minutes = max(1, int(diffs.median().total_seconds() // 60))
    return minutes


def _as_dataframe(price_bars: Iterable[PriceBar] | pd.DataFrame) -> pd.DataFrame:
    if isinstance(price_bars, pd.DataFrame):
        frame = price_bars.copy()
        if frame.index.tz is not None:
            frame.index = frame.index.tz_convert("UTC").tz_localize(None)
        frame = frame.sort_index()
        return frame
    rows: List[dict] = []
    for bar in price_bars:
        timestamp = bar.timestamp
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        timestamp = timestamp.astimezone(timezone.utc)
        rows.append(
            {
                "timestamp": pd.Timestamp(timestamp.replace(tzinfo=None)),
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": float(bar.volume),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    frame = pd.DataFrame(rows).set_index("timestamp").sort_index()
    return frame


@dataclass
class FeatureEngineerConfig:
    price_return_windows: Sequence[int] = (1, 5, 10, 20)
    volatility_windows: Sequence[int] = (10,)
    atr_window: int = 14
    moving_average_windows: Sequence[int] = (5, 10, 20)
    volume_zscore_window: int = 20
    sentiment_momentum_windows: Sequence[int] = (1, 3, 5)
    sentiment_zscore_window: int = 10
    label_horizon_minutes: int = 1440
    positive_threshold: float = 0.0035
    negative_threshold: float = -0.0035


class FeatureEngineer:
    def __init__(self, config: FeatureEngineerConfig | None = None) -> None:
        self.config = config or FeatureEngineerConfig()
        self._last_feature_columns: List[str] = []

    def build_training_frame(
        self,
        ticker: str,
        price_bars: Iterable[PriceBar] | pd.DataFrame,
        gdelt_window: Optional[GDELTWindow],
        *,
        include_labels: bool = True,
    ) -> pd.DataFrame:
        price_df = _as_dataframe(price_bars)
        if price_df.empty:
            return pd.DataFrame()
        price_df = price_df.sort_index()
        price_df = price_df.astype(float)

        enriched = self._price_features(price_df)
        enriched["session_date"] = enriched.index.normalize()

        sentiment_df = self._sentiment_features(gdelt_window, enriched["session_date"])
        enriched = enriched.join(sentiment_df, on="session_date", how="left")
        enriched = enriched.drop(columns=["session_date"])
        enriched = enriched.fillna(0.0)

        if include_labels:
            labeled = self._label_rows(enriched)
        else:
            labeled = enriched.copy()

        labeled["ticker"] = ticker
        labeled = labeled.replace({np.inf: 0.0, -np.inf: 0.0})
        if include_labels:
            labeled = labeled.dropna()
        else:
            labeled = labeled.dropna(how="all")

        self._last_feature_columns = self.feature_columns(labeled)
        return labeled

    def build_live_features(
        self,
        price_context: Iterable[PriceBar] | pd.DataFrame,
        gdelt_window: Optional[GDELTWindow],
    ) -> Optional[pd.Series]:
        frame = self.build_training_frame(
            "live",
            price_context,
            gdelt_window,
            include_labels=False,
        )
        if frame.empty:
            return None
        latest = frame.iloc[-1]
        return latest[self.feature_columns(frame)]

    def feature_columns(self, frame: Optional[pd.DataFrame] = None) -> List[str]:
        if frame is not None:
            excluded = {"label", "forward_return", "ticker"}
            return [col for col in frame.columns if col not in excluded]
        if self._last_feature_columns:
            return self._last_feature_columns
        return []

    def _price_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        df = frame.copy()
        df["open"] = df["open"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df["close"] = df["close"].astype(float)
        df["volume"] = df.get("volume", pd.Series(index=df.index, dtype=float)).fillna(0.0).astype(float)

        # Use fill_method=None to avoid future pandas deprecation warning
        df["return_1"] = df["close"].pct_change(fill_method=None).fillna(0.0)
        for window in self.config.price_return_windows:
            df[f"return_{window}"] = df["close"].pct_change(window, fill_method=None).fillna(0.0)

        for window in self.config.volatility_windows:
            df[f"volatility_{window}"] = df["return_1"].rolling(window, min_periods=2).std().fillna(0.0)

        df = self._atr_features(df)
        df = self._moving_average_features(df)
        df = self._volume_features(df)
        return df

    def _atr_features(self, df: pd.DataFrame) -> pd.DataFrame:
        high = df["high"]
        low = df["low"]
        close = df["close"]
        prev_close = close.shift(1)
        tr_components = pd.concat([
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1)
        true_range = tr_components.max(axis=1)
        atr = true_range.rolling(self.config.atr_window, min_periods=1).mean()
        df[f"atr_{self.config.atr_window}"] = atr.fillna(0.0)
        atr_pct = atr / close.replace({0.0: np.nan})
        df[f"atr_pct_{self.config.atr_window}"] = atr_pct.replace({np.nan: 0.0, np.inf: 0.0, -np.inf: 0.0})
        return df

    def _moving_average_features(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        for window in self.config.moving_average_windows:
            ma = close.rolling(window, min_periods=1).mean()
            df[f"sma_{window}"] = ma
            ratio = (close - ma) / ma.replace({0.0: np.nan})
            df[f"price_vs_sma_{window}"] = ratio.replace({np.nan: 0.0, np.inf: 0.0, -np.inf: 0.0})
        return df

    def _volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        volume = df["volume"]
        window = max(2, int(self.config.volume_zscore_window))
        rolling_mean = volume.rolling(window, min_periods=1).mean()
        rolling_std = volume.rolling(window, min_periods=1).std().replace({0.0: np.nan})
        zscore = (volume - rolling_mean) / rolling_std
        df["volume_z"] = zscore.replace({np.nan: 0.0, np.inf: 0.0, -np.inf: 0.0})
        return df

    def _sentiment_features(
        self,
        gdelt_window: Optional[GDELTWindow],
        session_dates: pd.Series,
    ) -> pd.DataFrame:
        unique_dates = pd.Index(sorted(session_dates.unique()), name="date")
        if gdelt_window is None or not gdelt_window.summaries:
            zeros = pd.DataFrame(index=unique_dates)
            return self._ensure_sentiment_columns(zeros)

        records = []
        for summary in gdelt_window.summaries:
            records.append(
                {
                    "date": pd.Timestamp(summary.date),
                    "sentiment_avg": float(summary.average_tone),
                    "sentiment_std": float(summary.tone_std),
                    "article_count": int(summary.article_count),
                    "positive_article_ratio": self._safe_ratio(
                        summary.positive_article_count,
                        summary.article_count,
                    ),
                    "negative_article_ratio": self._safe_ratio(
                        summary.negative_article_count,
                        summary.article_count,
                    ),
                    "timeline_points": int(summary.timeline_points),
                }
            )
        sentiment = pd.DataFrame(records)
        if sentiment.empty:
            sentiment = pd.DataFrame(index=unique_dates)
            return self._ensure_sentiment_columns(sentiment)
        sentiment = sentiment.sort_values("date").set_index("date")

        for window in self.config.sentiment_momentum_windows:
            shifted = sentiment["sentiment_avg"].shift(window)
            momentum = sentiment["sentiment_avg"] - shifted
            sentiment[f"sentiment_momentum_{window}"] = momentum.fillna(0.0)

        window = max(2, int(self.config.sentiment_zscore_window))
        rolling_mean = sentiment["sentiment_avg"].rolling(window, min_periods=1).mean()
        rolling_std = sentiment["sentiment_avg"].rolling(window, min_periods=1).std().replace({0.0: np.nan})
        zscore = (sentiment["sentiment_avg"] - rolling_mean) / rolling_std
        sentiment["sentiment_z"] = zscore.replace({np.nan: 0.0, np.inf: 0.0, -np.inf: 0.0})

        sentiment = sentiment.reindex(unique_dates).fillna(0.0)
        return self._ensure_sentiment_columns(sentiment)

    @staticmethod
    def _ensure_sentiment_columns(df: pd.DataFrame) -> pd.DataFrame:
        columns = [
            "sentiment_avg",
            "sentiment_std",
            "article_count",
            "positive_article_ratio",
            "negative_article_ratio",
            "timeline_points",
        ]
        momentum_cols = [col for col in df.columns if col.startswith("sentiment_momentum_")]
        columns.extend(momentum_cols)
        columns.append("sentiment_z")
        for column in columns:
            if column not in df.columns:
                df[column] = 0.0
        return df[sorted(set(columns))]

    @staticmethod
    def _safe_ratio(numerator: int, denominator: int) -> float:
        if denominator <= 0:
            return 0.0
        return float(numerator) / float(denominator)

    def _label_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        bar_minutes = _infer_bar_minutes(result.index)
        horizon_steps = max(1, int(self.config.label_horizon_minutes / max(bar_minutes, 1)))
        future_close = result["close"].shift(-horizon_steps)
        result["forward_return"] = (future_close - result["close"]) / result["close"]
        result["label"] = 0
        result.loc[result["forward_return"] >= self.config.positive_threshold, "label"] = 1
        result.loc[result["forward_return"] <= self.config.negative_threshold, "label"] = -1
        return result


__all__ = ["FeatureEngineer", "FeatureEngineerConfig"]
