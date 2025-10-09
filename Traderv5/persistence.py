"""Persistent storage helpers for TraderV5 runtime state."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Iterable, List, Optional, Tuple

import json

from Traderv4.funcs import RiskConfig


def _utcnow() -> datetime:
    return datetime.utcnow()


def _parse_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


@dataclass
class PositionPlan:
    """Persisted expectations for an open or pending position."""

    ticker: str
    side: str
    entry_price: float
    quantity: float
    target_price: float
    stop_loss_price: float
    expected_return_pct: float
    created_at: datetime
    last_seen: datetime
    expected_exit_at: Optional[datetime] = None
    metadata: Dict[str, Any] = None

    def as_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["created_at"] = self.created_at.isoformat()
        payload["last_seen"] = self.last_seen.isoformat()
        payload["expected_exit_at"] = (
            self.expected_exit_at.isoformat() if self.expected_exit_at else None
        )
        return payload

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PositionPlan":
        metadata = data.get("metadata") or {}
        return cls(
            ticker=str(data.get("ticker", "")).upper(),
            side=str(data.get("side", "")).upper() or "LONG",
            entry_price=float(data.get("entry_price", 0.0)),
            quantity=float(data.get("quantity", 0.0)),
            target_price=float(data.get("target_price", 0.0)),
            stop_loss_price=float(data.get("stop_loss_price", 0.0)),
            expected_return_pct=float(data.get("expected_return_pct", 0.0)),
            created_at=_parse_datetime(data.get("created_at")) or _utcnow(),
            last_seen=_parse_datetime(data.get("last_seen")) or _utcnow(),
            expected_exit_at=_parse_datetime(data.get("expected_exit_at")),
            metadata=dict(metadata),
        )


class PersistentTradeJournal:
    """Durable record of open positions and evaluation cadence."""

    def __init__(
        self,
        risk: RiskConfig,
        *,
        storage_path: Optional[Path] = None,
        variant: str = "core",
    ) -> None:
        self.risk = risk
        self.variant = variant
        self.storage_path = storage_path or Path("data/state/traderv5_state.json")
        self._lock = Lock()
        self._plans: Dict[str, PositionPlan] = {}
        self._last_prices: Dict[str, float] = {}
        self._last_evaluated: Dict[str, datetime] = {}
        self._history: List[Dict[str, Any]] = []
        self._dirty = False
        self._load()

    # ------------------------------------------------------------------
    # Persistence plumbing
    # ------------------------------------------------------------------
    def _load(self) -> None:
        path = self.storage_path
        try:
            if not path.exists():
                return
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return
        plans = payload.get("plans", {})
        history = payload.get("history", [])
        last_prices = payload.get("last_prices", {})
        last_evaluated = payload.get("last_evaluated", {})
        now = _utcnow()
        for ticker, data in plans.items():
            plan = PositionPlan.from_dict(data)
            if not plan.ticker:
                continue
            self._plans[plan.ticker] = plan
        self._history = list(history)[-250:]
        self._last_prices = {
            str(k).upper(): float(v)
            for k, v in last_prices.items()
            if isinstance(v, (float, int))
        }
        self._last_evaluated = {
            str(k).upper(): _parse_datetime(v) or now
            for k, v in last_evaluated.items()
        }

    def flush(self) -> None:
        with self._lock:
            if not self._dirty:
                return
            payload = {
                "variant": self.variant,
                "updated_at": _utcnow().isoformat(),
                "plans": {ticker: plan.as_dict() for ticker, plan in self._plans.items()},
                "history": self._history[-250:],
                "last_prices": self._last_prices,
                "last_evaluated": {
                    ticker: stamp.isoformat()
                    for ticker, stamp in self._last_evaluated.items()
                },
            }
            path = self.storage_path
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("w", encoding="utf-8") as handle:
                    json.dump(payload, handle, indent=2, sort_keys=True)
            except OSError:
                return
            self._dirty = False

    # ------------------------------------------------------------------
    # Scheduling helpers
    # ------------------------------------------------------------------
    def schedule(
        self,
        tickers: Iterable[str],
        *,
        now: Optional[datetime] = None,
        base_interval_seconds: int = 900,
        active_interval_seconds: Optional[int] = None,
    ) -> Tuple[List[str], int]:
        """Return tickers that need evaluation and the next wake-up interval."""

        now = now or _utcnow()
        active_interval = max(60, active_interval_seconds or max(60, base_interval_seconds // 3))
        due: List[str] = []
        soonest: Optional[int] = None
        for ticker in tickers:
            key = str(ticker).upper()
            plan = self._plans.get(key)
            interval = active_interval if plan else base_interval_seconds
            last = self._last_evaluated.get(key)
            if last is None:
                due.append(key)
                continue
            elapsed = (now - last).total_seconds()
            if elapsed >= interval:
                due.append(key)
            else:
                wait = int(max(5, interval - elapsed))
                if soonest is None or wait < soonest:
                    soonest = wait
        return due, int(soonest or max(base_interval_seconds, 60))

    def mark_evaluated(self, ticker: str, when: Optional[datetime] = None) -> None:
        with self._lock:
            self._last_evaluated[str(ticker).upper()] = when or _utcnow()
            self._dirty = True

    def update_market_price(self, ticker: str, price: float) -> None:
        if price <= 0:
            return
        with self._lock:
            self._last_prices[str(ticker).upper()] = float(price)
            self._dirty = True

    # ------------------------------------------------------------------
    # Position lifecycle
    # ------------------------------------------------------------------
    def plan_entry(
        self,
        ticker: str,
        *,
        side: str,
        entry_price: float,
        quantity: Optional[float],
        metadata: Optional[Dict[str, Any]] = None,
        expected_return_pct: Optional[float] = None,
    ) -> None:
        if entry_price <= 0:
            return
        key = str(ticker).upper()
        side = side.upper()
        take_pct, stop_pct = self._target_pcts(expected_return_pct)
        target_price, stop_price = self._project_prices(side, entry_price, take_pct, stop_pct)
        with self._lock:
            existing = self._plans.get(key)
            created_at = existing.created_at if existing else _utcnow()
            metadata_payload: Dict[str, Any] = {}
            if existing and existing.metadata:
                metadata_payload.update(existing.metadata)
            if metadata:
                metadata_payload.update(metadata)
            plan = PositionPlan(
                ticker=key,
                side=side,
                entry_price=entry_price,
                quantity=float(quantity) if quantity else (existing.quantity if existing else 0.0),
                target_price=target_price,
                stop_loss_price=stop_price,
                expected_return_pct=take_pct,
                created_at=created_at,
                last_seen=_utcnow(),
                expected_exit_at=self._expected_exit_time(created_at),
                metadata=metadata_payload,
            )
            self._plans[key] = plan
            self._dirty = True

    def sync_broker_positions(self, positions: Dict[str, Dict[str, Any]]) -> None:
        now = _utcnow()
        seen: set[str] = set()
        with self._lock:
            for raw_ticker, payload in positions.items():
                key = str(raw_ticker).upper()
                seen.add(key)
                qty = self._extract_float(payload, "qty", "quantity")
                side = str(payload.get("side", "")).upper() or "LONG"
                entry_price = self._extract_float(payload, "avg_entry_price", "avg_price")
                if entry_price <= 0 and qty > 0:
                    market_value = self._extract_float(payload, "market_value")
                    if market_value > 0:
                        entry_price = market_value / max(qty, 1e-6)
                take_pct, stop_pct = self._target_pcts()
                target_price, stop_price = self._project_prices(side, entry_price, take_pct, stop_pct)
                plan = self._plans.get(key)
                if plan is None:
                    plan = PositionPlan(
                        ticker=key,
                        side=side,
                        entry_price=entry_price,
                        quantity=qty,
                        target_price=target_price,
                        stop_loss_price=stop_price,
                        expected_return_pct=take_pct,
                        created_at=now,
                        last_seen=now,
                        expected_exit_at=self._expected_exit_time(now),
                        metadata={},
                    )
                    self._plans[key] = plan
                else:
                    plan.side = side
                    if entry_price > 0:
                        plan.entry_price = entry_price
                        plan.target_price, plan.stop_loss_price = self._project_prices(
                            side, entry_price, take_pct, stop_pct
                        )
                    plan.quantity = qty if qty > 0 else plan.quantity
                    plan.last_seen = now
                self._dirty = True
            for ticker in list(self._plans.keys()):
                if ticker not in seen:
                    plan = self._plans.pop(ticker)
                    self._history.append(
                        {
                            "ticker": plan.ticker,
                            "side": plan.side,
                            "closed_at": now.isoformat(),
                            "reason": "broker_position_closed",
                        }
                    )
                    self._dirty = True

    def confirm_exit(self, ticker: str, *, reason: str, price: Optional[float] = None) -> None:
        key = str(ticker).upper()
        with self._lock:
            plan = self._plans.pop(key, None)
            if plan:
                record = {
                    "ticker": key,
                    "side": plan.side,
                    "entry_price": plan.entry_price,
                    "target_price": plan.target_price,
                    "exit_price": price,
                    "reason": reason,
                    "created_at": plan.created_at.isoformat(),
                    "closed_at": _utcnow().isoformat(),
                }
                self._history.append(record)
            self._last_prices.pop(key, None)
            self._last_evaluated.pop(key, None)
            self._dirty = True

    def touch(self, ticker: str) -> None:
        with self._lock:
            plan = self._plans.get(str(ticker).upper())
            if plan:
                plan.last_seen = _utcnow()
                self._dirty = True

    def plan_for(self, ticker: str) -> Optional[PositionPlan]:
        return self._plans.get(str(ticker).upper())

    def active_positions(self) -> int:
        return len(self._plans)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _target_pcts(self, expected: Optional[float] = None) -> Tuple[float, float]:
        take_pct = float(expected) if expected is not None else float(self.risk.take_profit_pct)
        stop_pct = float(self.risk.stop_loss_pct)
        take_pct = max(0.0025, take_pct)
        stop_pct = max(0.0025, stop_pct)
        return take_pct, stop_pct

    def _project_prices(
        self,
        side: str,
        entry_price: float,
        take_pct: float,
        stop_pct: float,
    ) -> Tuple[float, float]:
        entry_price = float(entry_price)
        if entry_price <= 0:
            return 0.0, 0.0
        if side.upper() == "SHORT":
            target = entry_price * (1.0 - take_pct)
            stop = entry_price * (1.0 + stop_pct)
        else:
            target = entry_price * (1.0 + take_pct)
            stop = entry_price * (1.0 - stop_pct)
        return float(target), float(stop)

    def _expected_exit_time(self, start: datetime) -> datetime:
        return start + timedelta(minutes=max(int(self.risk.cooldown_minutes * 2), 60))

    @staticmethod
    def _extract_float(payload: Dict[str, Any], *keys: str) -> float:
        for key in keys:
            raw = payload.get(key)
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if value != 0:
                return value
        return 0.0


__all__ = ["PersistentTradeJournal", "PositionPlan"]
