"""Rate-limited HTTP client with filesystem caching."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import logging

import requests

_LOG = logging.getLogger("traderv5.http")


def _canonical_params(params: Optional[Mapping[str, Any]]) -> str:
    if not params:
        return ""
    items = sorted(params.items())
    return json.dumps(items, separators=(",", ":"), sort_keys=True)


def _cache_key(method: str, url: str, params: Optional[Mapping[str, Any]]) -> str:
    payload = f"{method.upper()}::{url}::{_canonical_params(params)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class CacheEntry:
    created_at: float
    ttl_seconds: float
    payload: Any

    def is_fresh(self, now: Optional[float] = None) -> bool:
        current = now or time.time()
        return (current - self.created_at) <= self.ttl_seconds


class ResponseCache:
    def __init__(self, cache_dir: Path, ttl: timedelta) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = max(ttl.total_seconds(), 1.0)
        self._lock = threading.Lock()

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def load(self, key: str, *, ttl_override: Optional[float] = None) -> Optional[CacheEntry]:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            try:
                path.unlink()
            except OSError:
                pass
            return None
        meta = payload.get("__meta__", {})
        created_at = float(meta.get("created_at", 0.0))
        ttl_seconds = float(meta.get("ttl_seconds", self.ttl_seconds))
        if ttl_override is not None:
            ttl_seconds = ttl_override
        entry = CacheEntry(created_at=created_at, ttl_seconds=ttl_seconds, payload=payload.get("data"))
        if entry.is_fresh():
            return entry
        try:
            path.unlink()
        except OSError:
            pass
        return None

    def store(self, key: str, data: Any, *, ttl_override: Optional[float] = None) -> None:
        path = self._path(key)
        record = {
            "__meta__": {
                "created_at": time.time(),
                "ttl_seconds": ttl_override if ttl_override is not None else self.ttl_seconds,
            },
            "data": data,
        }
        tmp_path = path.with_suffix(".tmp")
        with self._lock:
            with tmp_path.open("w", encoding="utf-8") as handle:
                json.dump(record, handle)
            tmp_path.replace(path)


class RateLimiter:
    def __init__(self, rate_per_second: float) -> None:
        if rate_per_second <= 0:
            raise ValueError("rate_per_second must be positive")
        self._min_interval = 1.0 / rate_per_second
        self._lock = threading.Lock()
        self._last_invocation = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_invocation
            remaining = self._min_interval - elapsed
            if remaining > 0:
                time.sleep(remaining)
                now = time.monotonic()
            self._last_invocation = now


class RateLimitedHttpClient:
    def __init__(
        self,
        *,
        queries_per_second: float,
        cache_dir: Path,
        cache_ttl: timedelta,
        max_retries: int = 3,
        backoff_factor: float = 1.5,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.session = session or requests.Session()
        self.rate_limiter = RateLimiter(queries_per_second)
        self.cache = ResponseCache(cache_dir, cache_ttl)
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

    def get_json(
        self,
        url: str,
        *,
        params: Optional[Mapping[str, Any]] = None,
        headers: Optional[Mapping[str, str]] = None,
        cache_key: Optional[str] = None,
        use_cache: bool = True,
        ttl_override: Optional[float] = None,
        timeout: float = 15.0,
    ) -> Any:
        key = cache_key or _cache_key("GET", url, params)
        if use_cache:
            cached = self.cache.load(key, ttl_override=ttl_override)
            if cached is not None:
                _LOG.debug("Returning cached response for %s", url)
                return cached.payload

        attempt = 0
        while True:
            attempt += 1
            self.rate_limiter.wait()
            try:
                response = self.session.get(url, params=params, headers=headers, timeout=timeout)
            except requests.RequestException as exc:
                if attempt <= self.max_retries:
                    sleep_time = self._backoff(attempt)
                    _LOG.debug("Request to %s failed with %s; retrying in %.2fs", url, exc, sleep_time)
                    time.sleep(sleep_time)
                    continue
                raise

            if response.status_code == 429 and attempt <= self.max_retries:
                sleep_time = self._backoff(attempt, throttled=True)
                _LOG.warning("Received 429 from %s; backing off %.2fs", url, sleep_time)
                time.sleep(sleep_time)
                continue
            try:
                response.raise_for_status()
            except requests.HTTPError:
                if attempt <= self.max_retries and response.status_code >= 500:
                    sleep_time = self._backoff(attempt)
                    _LOG.debug(
                        "Server error %s from %s; retrying in %.2fs",
                        response.status_code,
                        url,
                        sleep_time,
                    )
                    time.sleep(sleep_time)
                    continue
                raise

            try:
                payload = response.json()
            except ValueError:
                payload = response.text

            try:
                self.cache.store(key, payload, ttl_override=ttl_override)
            except Exception as exc:  # pragma: no cover - best effort cache
                _LOG.debug("Failed to cache response for %s: %s", url, exc)
            return payload

    def _backoff(self, attempt: int, throttled: bool = False) -> float:
        factor = self.backoff_factor if self.backoff_factor > 0 else 1.0
        base = factor if throttled else 0.5 * factor
        return min(60.0, base * (2 ** max(attempt - 1, 0)))


__all__ = [
    "RateLimitedHttpClient",
    "ResponseCache",
    "CacheEntry",
    "RateLimiter",
]
