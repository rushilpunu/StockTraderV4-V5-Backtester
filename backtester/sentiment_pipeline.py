"""Generate sentiment series for backtesting."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List

from .data_sources import fetch_gdelt_timeline


@dataclass
class SentimentPoint:
    timestamp: datetime
    average_sentiment: float
    sentiment_delta: float
    article_count: int


def build_sentiment_series(
    ticker: str,
    start: datetime,
    end: datetime,
    granularity_minutes: int = 60,
) -> List[SentimentPoint]:
    timeline = fetch_gdelt_timeline(ticker, start, end, minutes=granularity_minutes)
    points: List[SentimentPoint] = []
    previous_value: float = 0.0
    for timestamp, tone in timeline:
        delta = tone - previous_value
        points.append(
            SentimentPoint(
                timestamp=timestamp,
                average_sentiment=tone,
                sentiment_delta=delta,
                article_count=1,
            )
        )
        previous_value = tone
    return points


__all__ = ["SentimentPoint", "build_sentiment_series"]
