"""Advanced sentiment feature extraction for backtests."""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional, Tuple

import math
import warnings
import numpy as np
import pandas as pd

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore

    _vader = SentimentIntensityAnalyzer()
except Exception:  # pragma: no cover - fallback when vader not installed
    _vader = None

from .data_sources import fetch_gdelt_articles, fetch_gdelt_timeline, fetch_price_bars

_LOG = logging.getLogger(__name__)

try:
    from transformers import pipeline  # type: ignore
except Exception:  # pragma: no cover
    pipeline = None  # type: ignore

_finbert_pipeline = None

try:
    from keybert import KeyBERT  # type: ignore
except Exception:  # pragma: no cover
    KeyBERT = None  # type: ignore

_keybert_model = None

_SOURCE_WEIGHTS = defaultdict(
    lambda: 1.0,
    {
        "reuters.com": 1.3,
        "bloomberg.com": 1.3,
        "wsj.com": 1.25,
        "cnbc.com": 1.2,
        "finance.yahoo.com": 1.1,
        "seekingalpha.com": 1.1,
    },
)


@dataclass
class SentimentSnapshot:
    timestamp: datetime
    ticker: str
    tone_15: float
    tone_60: float
    tone_1440: float
    delta_15: float
    delta_60: float
    delta_1440: float
    vader: float
    finbert: Optional[float]
    keywords: List[str]
    source_score: float
    article_count: int


def _group_articles_by_time(raw_articles: Iterable[Dict], window: timedelta) -> Dict[datetime, List[Dict]]:
    buckets: Dict[datetime, List[Dict]] = defaultdict(list)
    for entry in raw_articles:
        seendate = entry.get("seendate") or entry.get("publishdate")
        if not seendate:
            continue
        try:
            timestamp = datetime.strptime(seendate, "%Y%m%dT%H%M%SZ")
        except ValueError:
            try:
                timestamp = datetime.strptime(seendate, "%Y%m%dT%H%M%S")
            except ValueError:
                continue
        bucket = timestamp - timedelta(minutes=timestamp.minute % window.seconds // 60, seconds=timestamp.second)
        buckets[bucket].append(entry)
    return buckets


def _compute_vader(texts: List[str], enabled: bool) -> float:
    if not texts or _vader is None or not enabled:
        return 0.0
    scores = [_vader.polarity_scores(text).get("compound", 0.0) for text in texts]
    return float(np.mean(scores)) if scores else 0.0


def _compute_finbert(texts: List[str], enabled: bool) -> Optional[float]:
    if not enabled or not texts:
        return None
    global _finbert_pipeline  # pylint: disable=global-statement
    if pipeline is None:
        _LOG.debug("Transformers pipeline unavailable; skipping FinBERT")
        return None
    if _finbert_pipeline is None:
        try:
            _finbert_pipeline = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                tokenizer="ProsusAI/finbert",
            )
        except Exception as exc:  # pragma: no cover
            _LOG.warning("FinBERT sentiment disabled: %s", exc)
            return None
    try:
        outputs = _finbert_pipeline(texts)
    except Exception as exc:  # pragma: no cover
        _LOG.warning("FinBERT inference failed: %s", exc)
        return None
    mapped = []
    for out in outputs:
        label = out.get("label", "neutral").lower()
        score = out.get("score", 0.0)
        if "positive" in label:
            mapped.append(score)
        elif "negative" in label:
            mapped.append(-score)
        else:
            mapped.append(0.0)
    return float(np.mean(mapped)) if mapped else None


def _extract_keywords(texts: List[str], enabled: bool) -> List[str]:
    if not texts or not enabled:
        return []

    cleaned = [text.strip() for text in texts if isinstance(text, str) and text.strip()]
    if not cleaned:
        return []

    blob = " ".join(cleaned)
    global _keybert_model
    if enabled and KeyBERT is not None and len(blob) > 120:
        if _keybert_model is None:
            try:
                _keybert_model = KeyBERT()
            except Exception as exc:  # pragma: no cover
                _LOG.warning("KeyBERT initialization failed: %s", exc)
                _keybert_model = None
        if _keybert_model is not None:
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("error", category=RuntimeWarning)
                    keywords = _keybert_model.extract_keywords(blob, top_n=5)
                extracted = [kw for kw, _ in keywords if kw]
                if extracted:
                    return extracted
            except (RuntimeWarning, ValueError):
                _LOG.debug("KeyBERT keyword extraction emitted invalid values; falling back", exc_info=True)
            except Exception:  # pragma: no cover
                _LOG.debug("KeyBERT extraction failed; falling back", exc_info=True)

    words = [word.lower() for text in cleaned for word in text.split() if len(word) > 3]
    most_common = [word for word, _ in Counter(words).most_common(5)]
    return most_common


def _timeline_dict(timeline: List[Tuple[datetime, float]]) -> Dict[pd.Timestamp, float]:
    return {pd.Timestamp(ts).tz_localize(None): val for ts, val in timeline}


def _value_at(ts: pd.Timestamp, mapping: Dict[pd.Timestamp, float]) -> float:
    if ts in mapping:
        return mapping[ts]
    before = [time for time in mapping if time <= ts]
    if not before:
        return 0.0
    nearest = max(before)
    return mapping[nearest]


def _price_derived_snapshots(
    ticker: str,
    start: datetime,
    end: datetime,
    granularity_minutes: int,
) -> List[SentimentSnapshot]:
    """Derive sentiment-like signals directly from historical prices."""

    # Expand the lookback window so momentum calculations have context.
    lookback_days = max(10, granularity_minutes // 60 * 5)
    start_buffer = start - timedelta(days=lookback_days)
    try:
        price_df = fetch_price_bars(ticker, start_buffer, end, timeframe="1Day")
    except Exception as exc:  # pragma: no cover - defensive logging
        _LOG.error("Unable to derive price-based sentiment for %s: %s", ticker, exc)
        return []
    if price_df.empty or "close" not in price_df.columns:
        return []

    frame = price_df.copy()
    frame.index = pd.to_datetime(frame.index)
    frame = frame.sort_index()
    closes = frame["close"].astype(float)
    opens = frame.get("open", closes)
    volumes = frame.get("volume", pd.Series(0.0, index=frame.index)).astype(float)

    returns = closes.pct_change().fillna(0.0)
    tone_short = returns.rolling(3, min_periods=1).mean()
    tone_medium = returns.rolling(10, min_periods=1).mean()
    tone_long = returns.rolling(21, min_periods=1).mean()

    range_ratio = ((closes - opens) / opens.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    volume_ratio = (volumes / volumes.rolling(20, min_periods=1).mean()).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    prev_closes = closes.shift(1).bfill().fillna(closes)

    snapshots: List[SentimentSnapshot] = []
    prev_short = prev_medium = prev_long = 0.0
    for ts, short_val, medium_val, long_val in zip(tone_short.index, tone_short, tone_medium, tone_long):
        if ts < pd.Timestamp(start) or ts > pd.Timestamp(end):
            continue
        vader_score = float(np.tanh(range_ratio.loc[ts] * 5.0))
        vol_ratio = float(np.clip(volume_ratio.loc[ts], 0.0, 5.0))
        movement = float(closes.loc[ts] - prev_closes.loc[ts])
        keywords = []
        if movement > 0:
            keywords.append("momentum")
        elif movement < 0:
            keywords.append("pullback")
        else:
            keywords.append("range-bound")
        if abs(short_val) > abs(medium_val):
            keywords.append("impulse")
        else:
            keywords.append("trend")
        keywords = list(dict.fromkeys(keywords))
        article_count = max(0, int(round(vol_ratio * 2)))

        snapshots.append(
            SentimentSnapshot(
                timestamp=ts.to_pydatetime(),
                ticker=ticker,
                tone_15=float(short_val),
                tone_60=float(medium_val),
                tone_1440=float(long_val),
                delta_15=float(short_val - prev_short),
                delta_60=float(medium_val - prev_medium),
                delta_1440=float(long_val - prev_long),
                vader=vader_score,
                finbert=None,
                keywords=keywords,
                source_score=vol_ratio,
                article_count=article_count,
            )
        )
        prev_short, prev_medium, prev_long = short_val, medium_val, long_val

    return snapshots


def build_sentiment_snapshots(
    ticker: str,
    start: datetime,
    end: datetime,
    granularity_minutes: int = 60,
    *,
    use_vader: bool = True,
    use_finbert: bool = False,
    use_keybert: bool = False,
) -> List[SentimentSnapshot]:
    raw_articles = fetch_gdelt_articles(ticker, start, end)
    article_buckets = _group_articles_by_time(raw_articles, timedelta(minutes=granularity_minutes))
    timeline_15 = fetch_gdelt_timeline(ticker, start, end, minutes=15)
    timeline_60 = fetch_gdelt_timeline(ticker, start, end, minutes=60)
    timeline_1440 = fetch_gdelt_timeline(ticker, start, end, minutes=1440)

    import pandas as pd

    map15 = _timeline_dict(timeline_15)
    map60 = _timeline_dict(timeline_60)
    map1440 = _timeline_dict(timeline_1440)

    timestamps = sorted(map15.keys())
    snapshots: List[SentimentSnapshot] = []
    prev15 = prev60 = prev1440 = 0.0
    for ts in timestamps:
        tone15 = _value_at(ts, map15)
        tone60 = _value_at(ts, map60)
        tone1440 = _value_at(ts, map1440)
        delta15 = tone15 - prev15
        delta60 = tone60 - prev60
        delta1440 = tone1440 - prev1440

        bucket_key = ts.to_pydatetime().replace(second=0, microsecond=0)
        articles = article_buckets.get(bucket_key, [])
        texts = [entry.get("title", "") for entry in articles]
        vader_score = _compute_vader(texts, enabled=use_vader)
        finbert_score = _compute_finbert(texts, enabled=use_finbert)
        keywords = _extract_keywords(texts, enabled=use_keybert)
        source_score = float(np.mean([_SOURCE_WEIGHTS[entry.get("domain", "")] for entry in articles])) if articles else 0.0

        snapshots.append(
            SentimentSnapshot(
                timestamp=ts.to_pydatetime(),
                ticker=ticker,
                tone_15=tone15,
                tone_60=tone60,
                tone_1440=tone1440,
                delta_15=delta15,
                delta_60=delta60,
                delta_1440=delta1440,
                vader=vader_score,
                finbert=finbert_score,
                keywords=keywords,
                source_score=source_score,
                article_count=len(articles),
            )
        )

        prev15, prev60, prev1440 = tone15, tone60, tone1440
    if not snapshots:
        snapshots = _price_derived_snapshots(ticker, start, end, granularity_minutes)
    return snapshots


__all__ = ["SentimentSnapshot", "build_sentiment_snapshots"]
