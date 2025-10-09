"""Persistent storage helpers for TraderV5 runtime state."""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

_LOG = logging.getLogger("traderv5.persistence")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _to_iso(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    return dt.astimezone(timezone.utc).isoformat()


def _from_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        _LOG.debug("Unable to parse stored datetime '%s'", value)
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


@dataclass
class StoredPosition:
    """Represents the intent the model had when entering a position."""

    ticker: str
    side: str
    quantity: float
    entry_price: float
    target_price: float
    stop_price: Optional[float] = None
    entry_time: datetime = field(default_factory=_utc_now)
    expected_exit: Optional[datetime] = None
    tolerance: float = 0.10
    pending_exit: bool = False
    last_price: Optional[float] = None
    last_updated: datetime = field(default_factory=_utc_now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "ticker": self.ticker,
            "side": self.side,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "target_price": self.target_price,
            "stop_price": self.stop_price,
            "entry_time": _to_iso(self.entry_time),
            "expected_exit": _to_iso(self.expected_exit),
            "tolerance": self.tolerance,
            "pending_exit": self.pending_exit,
            "last_price": self.last_price,
            "last_updated": _to_iso(self.last_updated),
            "metadata": self.metadata,
        }
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StoredPosition":
        return cls(
            ticker=str(payload.get("ticker", "")).upper(),
            side=str(payload.get("side", "")).upper(),
            quantity=float(payload.get("quantity", 0.0)),
            entry_price=float(payload.get("entry_price", 0.0)),
            target_price=float(payload.get("target_price", 0.0)),
            stop_price=(
                float(payload["stop_price"])
                if payload.get("stop_price") is not None
                else None
            ),
            entry_time=_from_iso(payload.get("entry_time")) or _utc_now(),
            expected_exit=_from_iso(payload.get("expected_exit")),
            tolerance=float(payload.get("tolerance", 0.10)),
            pending_exit=bool(payload.get("pending_exit", False)),
            last_price=(
                float(payload["last_price"])
                if payload.get("last_price") is not None
                else None
            ),
            last_updated=_from_iso(payload.get("last_updated")) or _utc_now(),
            metadata=dict(payload.get("metadata", {})),
        )

    def copy(self) -> "StoredPosition":
        return StoredPosition(
            ticker=self.ticker,
            side=self.side,
            quantity=self.quantity,
            entry_price=self.entry_price,
            target_price=self.target_price,
            stop_price=self.stop_price,
            entry_time=self.entry_time,
            expected_exit=self.expected_exit,
            tolerance=self.tolerance,
            pending_exit=self.pending_exit,
            last_price=self.last_price,
            last_updated=self.last_updated,
            metadata=dict(self.metadata),
        )


class PositionPersistence:
    """Thread-safe JSON-backed persistence for position intent."""

    def __init__(self, path: Path, *, autosave: bool = True) -> None:
        self.path = path.expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.autosave = autosave
        self._lock = threading.Lock()
        self._positions: Dict[str, StoredPosition] = {}
        self._load()

    @classmethod
    def for_variant(cls, name: str) -> "PositionPersistence":
        safe = name.strip().lower().replace(" ", "_") or "core"
        base = Path("data/state")
        return cls(base / f"{safe}_positions.json")

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                raw = json.load(handle)
        except Exception as exc:  # pragma: no cover - IO guard
            _LOG.warning("Unable to load position store %s: %s", self.path, exc)
            return
        if not isinstance(raw, list):
            _LOG.warning("Unexpected payload in %s; expected list", self.path)
            return
        for item in raw:
            try:
                record = StoredPosition.from_dict(item)
            except Exception as exc:  # pragma: no cover - guard
                _LOG.debug("Skipping malformed position entry: %s", exc)
                continue
            if record.ticker:
                self._positions[record.ticker] = record

    def _save_locked(self) -> None:
        if not self.autosave:
            return
        payload = [record.to_dict() for record in self._positions.values()]
        with self.path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)

    def get(self, ticker: str) -> Optional[StoredPosition]:
        key = ticker.upper()
        with self._lock:
            record = self._positions.get(key)
            return record.copy() if record else None

    def upsert(self, record: StoredPosition) -> None:
        with self._lock:
            self._positions[record.ticker] = record.copy()
            self._save_locked()

    def remove(self, ticker: str) -> None:
        key = ticker.upper()
        with self._lock:
            if key in self._positions:
                del self._positions[key]
                self._save_locked()

    def all(self) -> List[StoredPosition]:
        with self._lock:
            return [item.copy() for item in self._positions.values()]

    def mark_pending_exit(self, ticker: str) -> None:
        key = ticker.upper()
        with self._lock:
            record = self._positions.get(key)
            if record is None:
                return
            record.pending_exit = True
            record.last_updated = _utc_now()
            self._positions[key] = record
            self._save_locked()

    def touch(self, ticker: str, *, price: Optional[float] = None) -> None:
        key = ticker.upper()
        with self._lock:
            record = self._positions.get(key)
            if record is None:
                return
            if price is not None:
                record.last_price = price
            record.last_updated = _utc_now()
            self._positions[key] = record
            self._save_locked()

    def reconcile(self, broker_positions: Iterable[Mapping[str, Any]]) -> None:
        seen: Dict[str, Mapping[str, Any]] = {}
        for payload in broker_positions:
            ticker_raw = payload.get("ticker") or payload.get("symbol")
            if not ticker_raw:
                continue
            key = str(ticker_raw).upper()
            seen[key] = payload

        with self._lock:
            now = _utc_now()
            for ticker, payload in seen.items():
                record = self._positions.get(ticker)
                if record is None:
                    continue
                try:
                    record.quantity = float(payload.get("quantity", record.quantity))
                except Exception:
                    pass
                side_raw = payload.get("side")
                if side_raw:
                    record.side = str(side_raw).upper()
                price_raw = payload.get("current_price") or payload.get("market_price")
                if price_raw is None:
                    market_value = payload.get("market_value")
                    qty = record.quantity or payload.get("quantity")
                    try:
                        qty_f = float(qty)
                        if qty_f:
                            price_raw = float(market_value) / qty_f
                    except Exception:
                        price_raw = None
                if price_raw is not None:
                    try:
                        record.last_price = float(price_raw)
                    except Exception:
                        record.last_price = record.last_price
                record.pending_exit = record.pending_exit and record.quantity > 0
                record.last_updated = now
                self._positions[ticker] = record

            for ticker in list(self._positions.keys()):
                if ticker not in seen:
                    del self._positions[ticker]

            self._save_locked()


__all__ = ["StoredPosition", "PositionPersistence"]

