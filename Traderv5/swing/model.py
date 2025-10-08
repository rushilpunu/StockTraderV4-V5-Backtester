"""Swing-trading specific heuristic model for TraderV5."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, MutableMapping, Sequence, Union

import math
import statistics

import numpy as np
import pandas as pd


FeatureLike = Union[Mapping[str, float], pd.Series, np.ndarray, Sequence[float]]


@dataclass(frozen=True)
class SwingModelConfig:
    """Configuration for the handcrafted swing predictor."""

    long_bias: float = 0.18
    momentum_weight: float = 0.48
    trend_weight: float = 0.42
    sentiment_weight: float = 0.22
    volatility_weight: float = 0.35
    volume_weight: float = 0.12
    article_weight: float = 0.08
    cooldown_decay: float = 0.82
    swing_horizon_days: int = 5


class SwingModelPredictor:
    """Rule-based probability estimator tuned for multi-day swings.

    The model is intentionally conservative – it only surfaces a high
    probability when trend, momentum, and sentiment all line up in the same
    direction. This keeps the trade frequency low but focuses entries on the
    stronger follow-through environments required for swing trading.
    """

    def __init__(self, config: SwingModelConfig | None = None) -> None:
        self.config = config or SwingModelConfig()
        self._feature_history: MutableMapping[str, list[float]] = {}

    def predict_proba(self, features: FeatureLike) -> float:
        as_dict = self._normalise_features(features)
        cfg = self.config

        price_vs_sma_20 = as_dict.get("price_vs_sma_20", 0.0)
        price_vs_sma_10 = as_dict.get("price_vs_sma_10", 0.0)
        return_5 = as_dict.get("return_5", 0.0)
        return_10 = as_dict.get("return_10", 0.0)
        return_20 = as_dict.get("return_20", 0.0)
        atr_pct = as_dict.get("atr_pct_14", 0.0)
        volatility_10 = as_dict.get("volatility_10", 0.0)
        volume_z = as_dict.get("volume_z", 0.0)
        sentiment_avg = as_dict.get("sentiment_avg", 0.0)
        sentiment_z = as_dict.get("sentiment_z", 0.0)
        pos_ratio = as_dict.get("positive_article_ratio", 0.0)
        neg_ratio = as_dict.get("negative_article_ratio", 0.0)
        article_count = as_dict.get("article_count", 0.0)

        # Track rolling conviction to avoid over-trading the same name.
        history_key = "trend_signal"
        composite_trend = 0.6 * price_vs_sma_20 + 0.4 * price_vs_sma_10
        trend_history = self._feature_history.setdefault(history_key, [])
        trend_history.append(composite_trend)
        if len(trend_history) > 32:
            del trend_history[0]
        median_trend = statistics.median(trend_history) if trend_history else 0.0

        momentum_raw = 0.55 * return_10 + 0.45 * return_20
        momentum_score = max(0.0, momentum_raw)
        trend_raw = 0.7 * composite_trend + 0.3 * median_trend
        trend_score = max(0.0, trend_raw)

        sentiment_score = (
            sentiment_avg * 0.65
            + sentiment_z * 0.25
            + (pos_ratio - neg_ratio) * 0.35
        )

        volatility_floor = max(0.0, atr_pct - 0.045)
        volatility_penalty = cfg.volatility_weight * volatility_floor
        noise_penalty = cfg.volatility_weight * max(0.0, volatility_10 - 0.025)

        volume_boost = cfg.volume_weight * max(0.0, min(3.0, volume_z))
        article_boost = cfg.article_weight * math.log1p(max(0.0, article_count - 1.0))

        bearish_penalty = 0.0
        if return_5 < 0:
            bearish_penalty += abs(return_5) * 0.8
        if return_10 < 0:
            bearish_penalty += abs(return_10) * 0.65
        if price_vs_sma_20 < 0:
            bearish_penalty += abs(price_vs_sma_20) * 0.5

        raw_signal = (
            cfg.long_bias
            + cfg.momentum_weight * momentum_score
            + cfg.trend_weight * trend_score
            + cfg.sentiment_weight * sentiment_score
            + volume_boost
            + article_boost
            - volatility_penalty
            - noise_penalty
            - bearish_penalty
        )

        # Sharpen the signal through a logistic transform.
        probability = 1.0 / (1.0 + math.exp(-3.6 * (raw_signal - 0.12)))

        # Apply additional damping if the recent composite trend has flipped.
        if trend_history and composite_trend < 0:
            probability *= 0.62
        elif trend_history and trend_history[-1] < composite_trend:
            probability *= 1.05

        # Cap the probability to avoid unrealistic confidence.
        probability = float(max(0.02, min(0.985, probability)))
        return probability

    def predict_class(self, features: FeatureLike) -> int:
        return 1 if self.predict_proba(features) >= 0.5 else -1

    def _normalise_features(self, features: FeatureLike) -> Dict[str, float]:
        if isinstance(features, pd.Series):
            result: Dict[str, float] = {}
            for key in features.index:
                value = features.get(key, 0.0)
                try:
                    result[key] = float(value)
                except (TypeError, ValueError):
                    continue
            return result
        if isinstance(features, Mapping):
            result: Dict[str, float] = {}
            for key, value in features.items():
                try:
                    result[key] = float(value)
                except (TypeError, ValueError):
                    continue
            return result
        if isinstance(features, np.ndarray):
            raise TypeError(
                "SwingModelPredictor expects dict-like features; received numpy array."
            )
        if isinstance(features, Iterable):
            raise TypeError(
                "SwingModelPredictor expects named features; received positional sequence."
            )
        raise TypeError(f"Unsupported feature payload: {type(features)!r}")


def load_swing_predictor() -> SwingModelPredictor:
    """Factory helper used by entry points to load the swing model."""

    return SwingModelPredictor()


__all__ = ["SwingModelPredictor", "SwingModelConfig", "load_swing_predictor"]
