import json
import logging
import os
import re
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional

import requests

try:
    import fcntl  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None


@dataclass
class GDELTArticle:
    title: str
    url: str
    published_at: datetime
    tone: float
    ticker: Optional[str]


class GDELTClient:
    BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
    _TICKER_PATTERN = re.compile(r"^[A-Z]{1,5}(?:\.[A-Z]{1,2})?$")
    _REQUEST_COUNTER = 0
    _REQUEST_LOCK = Lock()
    _RATE_FILE = Path(tempfile.gettempdir()) / "gdelt_api.lock"
    _LOGGER = logging.getLogger("gdelt_client")

    def __init__(self, delay: float = 1.5):
        self.delay = delay

    @classmethod
    def _next_request_id(cls) -> int:
        with cls._REQUEST_LOCK:
            cls._REQUEST_COUNTER += 1
            return cls._REQUEST_COUNTER

    @contextmanager
    def _rate_guard(self):
        if fcntl is None:
            yield
            return
        handle = open(self._RATE_FILE, "w")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
            handle.close()

    def _log_rate_limit(self, request_id: int, attempt: int, max_attempts: int) -> None:
        self._LOGGER.warning(
            "GDELT rate limit encountered; being limited, on %s request of %s total requests (request #%s)",
            attempt,
            max_attempts,
            request_id,
        )

    def _request(self, params: Dict[str, str], *, max_attempts: int = 5) -> Dict:
        request_id = self._next_request_id()
        last_error: Optional[RuntimeError] = None
        for attempt in range(1, max_attempts + 1):
            with self._rate_guard():
                response = requests.get(self.BASE_URL, params=params, timeout=10)
                if response.status_code == 429:
                    self._log_rate_limit(request_id, attempt, max_attempts)
                    backoff = max(5.0, self.delay * (attempt + 1))
                    time.sleep(backoff)
                    continue
                if response.status_code != 200:
                    last_error = RuntimeError(f"GDELT API error {response.status_code}: {response.text}")
                    break
                try:
                    payload = response.json()
                except json.JSONDecodeError as exc:
                    last_error = RuntimeError("Failed to decode JSON response")
                    break
                time.sleep(self.delay)
                return payload
        if last_error is not None:
            raise last_error
        raise RuntimeError(
            f"GDELT API error 429: exceeded retry budget for request #{request_id}"
        )

    def _resolve_ticker(self, entry: Dict[str, str], fallback: str) -> str:
        fallback = fallback.upper()
        candidates = [
            entry.get("semtag"),
            entry.get("ticker"),
            entry.get("symbol"),
            entry.get("themes"),
        ]
        for raw in candidates:
            if not raw:
                continue
            candidate = raw.split(",")[0]
            if ":" in candidate:
                candidate = candidate.split(":")[-1]
            candidate = candidate.strip().upper()
            if not candidate:
                continue
            if self._TICKER_PATTERN.match(candidate):
                return candidate
        return fallback

    def fetch_recent_articles(self, ticker: str, minutes_back: int = 90) -> List[GDELTArticle]:
        timespan = f"{minutes_back}min"
        payload = self._request({
            "query": ticker,
            "mode": "ArtList",
            "format": "JSON",
            "maxrecords": "250",
            "sort": "DateDesc",
            "timespan": timespan,
        })
        records: List[GDELTArticle] = []
        for entry in payload.get("articles", []):
            published_raw = entry.get("seendate") or entry.get("publishdate") or entry.get("date")
            published_at = self._parse_timestamp(published_raw)
            ticker_tag = self._resolve_ticker(entry, ticker)
            tone = float(entry.get("tone", 0.0))
            records.append(GDELTArticle(
                title=entry.get("title", ""),
                url=entry.get("sourceurl", ""),
                published_at=published_at,
                tone=tone,
                ticker=ticker_tag,
            ))
        return records

    def fetch_sentiment_timeline(self, ticker: str, minutes_back: int = 180) -> Dict:
        timespan = f"{minutes_back}min"
        return self._request({
            "query": ticker,
            "mode": "TimelineTone",
            "format": "JSON",
            "timespan": timespan,
        })

    def _parse_timestamp(self, raw: Optional[str]) -> datetime:
        if not raw:
            return datetime.utcnow()
        formats = [
            "%Y%m%dT%H%M%S",
            "%Y%m%d%H%M%S",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%d %H:%M:%S",
        ]
        for fmt in formats:
            try:
                return datetime.strptime(raw, fmt)
            except ValueError:
                continue
        if raw.endswith("Z"):
            with_z = raw[:-1]
            for fmt in ["%Y%m%dT%H%M%S", "%Y-%m-%dT%H:%M:%S"]:
                try:
                    return datetime.strptime(with_z, fmt)
                except ValueError:
                    continue
        return datetime.utcnow()


if __name__ == "__main__":
    client = GDELTClient()
    print(client.fetch_recent_articles("AAPL", minutes_back=60))
