"""Model-driven decision engine for Trader V5."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, Optional, Tuple

import math

from Traderv4.funcs import (
    AccountSnapshot,
    RiskConfig,
    TradeDecision,
    TradeFrequencyTracker,
)

from Traderv5.model.predictor import ModelPredictor
from Traderv5.persistence import PositionPersistence, StoredPosition


@dataclass
class SignalContext:
    probability_long: float
    features: Dict[str, float]

    @property
    def signal_strength(self) -> float:
        return float(self.probability_long - 0.5)


@dataclass(frozen=True)
class ExitPlan:
    target_price: float
    stop_price: Optional[float]
    tolerance: float
    expected_exit: Optional[datetime]


class ModelDecisionEngine:
    def __init__(
        self,
        predictor: ModelPredictor,
        risk: RiskConfig,
        trade_tracker: TradeFrequencyTracker,
        position_store: Optional[PositionPersistence] = None,
    ) -> None:
        self.predictor = predictor
        self.risk = risk
        self.trade_tracker = trade_tracker
        self.last_trade_at: Dict[str, datetime] = {}
        self.position_store = position_store
        self._session_memory: Dict[str, StoredPosition] = {}

    def decide(
        self,
        ticker: str,
        features: Dict[str, float],
        price_snapshot: Dict[str, Optional[float]],
        account: AccountSnapshot,
        *,
        position: Optional[Dict[str, Any]] = None,
        open_positions: int = 0,
    ) -> TradeDecision:
        price_raw = price_snapshot.get("close") if price_snapshot else None
        try:
            price = float(price_raw) if price_raw is not None else 0.0
        except (TypeError, ValueError):
            price = 0.0
        if price <= 0:
            return self._hold(ticker, "missing price data")

        probability = float(self.predictor.predict_proba(features))
        signal = SignalContext(probability_long=probability, features=features)
        strength = signal.signal_strength

        def _feature(name: str, default: float = 0.0) -> float:
            raw_value = features.get(name, default)
            try:
                return float(raw_value)
            except (TypeError, ValueError):
                return default

        aggressiveness = max(0.25, float(getattr(self.risk, "aggressiveness", 1.0)))
        entry_bias = float(getattr(self.risk, "entry_signal_bias", 0.0))
        base_entry_threshold = max(0.0, float(self.risk.entry_sentiment_threshold) - entry_bias)
        strength_threshold = max(0.02, min(0.35, base_entry_threshold / aggressiveness))

        exit_base = max(0.0, float(self.risk.exit_sentiment_threshold) - entry_bias * 0.5)
        exit_threshold = max(0.01, min(0.25, exit_base / max(1.0, aggressiveness * 0.7)))
        long_exit_threshold = exit_threshold
        short_exit_threshold = -exit_threshold

        pos_side, quantity, market_value = self._parse_position(position, price)
        memory = self._memory_for(ticker)
        if position and memory:
            self._sync_memory_with_broker(ticker, memory, position, price)

        available_funds, per_position_cap = self._position_budget(account, market_value)

        momentum = (
            0.35 * _feature("return_5")
            + 0.25 * _feature("price_vs_sma_10")
            + 0.2 * _feature("return_10")
            + 0.2 * _feature("price_vs_sma_20")
        )
        short_momentum = (
            0.45 * _feature("return_5")
            + 0.35 * _feature("return_10")
            + 0.2 * _feature("price_vs_sma_20")
        )
        sentiment_avg = _feature("sentiment_avg")
        sentiment_std = _feature("sentiment_std")
        volume_z = _feature("volume_z")
        volatility_pct = abs(_feature("atr_pct_14"))
        return_1 = _feature("return_1")
        return_20 = _feature("return_20")
        price_vs_sma_5 = _feature("price_vs_sma_5")
        price_vs_sma_10 = _feature("price_vs_sma_10")
        price_vs_sma_20 = _feature("price_vs_sma_20")

        momentum_floor = getattr(self.risk, "momentum_entry_floor", 0.0015)
        quick_floor = getattr(self.risk, "fast_momentum_floor", momentum_floor * 0.5)
        sentiment_floor = getattr(self.risk, "sentiment_entry_floor", -0.05)
        volume_floor = getattr(self.risk, "volume_entry_floor", -1.25)
        volatility_cap = getattr(self.risk, "volatility_entry_cap", 0.14)
        price_bias_floor = getattr(self.risk, "price_bias_entry_floor", -0.01)

        short_momentum_ceiling = getattr(self.risk, "momentum_short_ceiling", -momentum_floor)
        short_sentiment_ceiling = getattr(self.risk, "sentiment_short_ceiling", 0.08)
        short_volume_floor = getattr(self.risk, "volume_short_floor", -1.5)
        short_price_ceiling = getattr(self.risk, "price_bias_short_ceiling", 0.02)
        short_volatility_cap = getattr(
            self.risk,
            "volatility_short_cap",
            volatility_cap * 1.3 if volatility_cap > 0 else 0.0,
        )
        short_fast_threshold = short_momentum_ceiling * 0.5
        volatility_cap_str = "n/a" if volatility_cap <= 0 else f"{volatility_cap:.3f}"
        short_volatility_cap_str = "n/a" if short_volatility_cap <= 0 else f"{short_volatility_cap:.3f}"

        long_gates = {
            "momentum": momentum >= momentum_floor,
            "fast_return": return_1 >= quick_floor,
            "sentiment": sentiment_avg >= sentiment_floor,
            "volume": volume_z >= volume_floor,
            "volatility": True if volatility_cap <= 0 else volatility_pct <= volatility_cap,
            "price_bias": price_vs_sma_10 >= price_bias_floor and price_vs_sma_20 >= price_bias_floor,
        }
        short_gates = {
            "momentum": short_momentum <= short_momentum_ceiling,
            "fast_return": return_1 <= short_momentum_ceiling * 0.5,
            "sentiment": sentiment_avg <= short_sentiment_ceiling,
            "volume": volume_z >= short_volume_floor,
            "volatility": True if short_volatility_cap <= 0 else volatility_pct <= short_volatility_cap,
            "price_bias": price_vs_sma_10 <= short_price_ceiling and price_vs_sma_20 <= short_price_ceiling,
        }

        def _gate_summary(gates: Dict[str, bool]) -> str:
            return ",".join(f"{name}:{int(flag)}" for name, flag in gates.items())

        base_metadata: Dict[str, str] = {
            "prob_long": f"{probability:.3f}",
            "signal_strength": f"{strength:.3f}",
            "entry_threshold": f"{strength_threshold:.3f}",
            "exit_threshold": f"{exit_threshold:.3f}",
            "momentum": f"{momentum:.4f}",
            "short_momentum": f"{short_momentum:.4f}",
            "sentiment": f"{sentiment_avg:.4f}",
            "sentiment_std": f"{sentiment_std:.4f}",
            "volume_z": f"{volume_z:.3f}",
            "volatility_pct": f"{volatility_pct:.4f}",
            "return_1": f"{return_1:.4f}",
            "return_20": f"{return_20:.4f}",
            "price_vs_sma_5": f"{price_vs_sma_5:.4f}",
            "price_vs_sma_10": f"{price_vs_sma_10:.4f}",
            "price_vs_sma_20": f"{price_vs_sma_20:.4f}",
            "available_funds": f"{available_funds:.2f}",
            "per_position_cap": f"{per_position_cap:.2f}",
            "long_gates": _gate_summary(long_gates),
            "short_gates": _gate_summary(short_gates),
            "long_gate_thresholds": (
                f"momentum>={momentum_floor:.4f};fast>={quick_floor:.4f};sentiment>={sentiment_floor:.4f};"
                f"volume>={volume_floor:.2f};volatility<={volatility_cap_str};price_bias>={price_bias_floor:.4f}"
            ),
            "short_gate_thresholds": (
                f"momentum<={short_momentum_ceiling:.4f};fast<={short_fast_threshold:.4f};sentiment<={short_sentiment_ceiling:.4f};"
                f"volume>={short_volume_floor:.2f};volatility<={short_volatility_cap_str};price_bias<={short_price_ceiling:.4f}"
            ),
        }

        # Exit logic for open positions
        score_note = {
            "probability_long": base_metadata["prob_long"],
            "signal_strength": base_metadata["signal_strength"],
            "momentum": base_metadata["momentum"],
            "sentiment": base_metadata["sentiment"],
        }

        if pos_side == "LONG" and quantity > 0:
            exit_decision = self._evaluate_exit(
                ticker,
                side="LONG",
                quantity=quantity,
                price=price,
                market_value=market_value,
                strength=strength,
                gate_snapshot=base_metadata["long_gates"],
                threshold=strength_threshold,
                exit_threshold=long_exit_threshold,
                score_note=score_note,
                memory=memory,
            )
            if exit_decision:
                return exit_decision
            return self._hold(ticker, "maintain long position", metadata=dict(base_metadata))

        if pos_side == "SHORT" and quantity > 0:
            exit_decision = self._evaluate_exit(
                ticker,
                side="SHORT",
                quantity=quantity,
                price=price,
                market_value=market_value,
                strength=strength,
                gate_snapshot=base_metadata["short_gates"],
                threshold=strength_threshold,
                exit_threshold=short_exit_threshold,
                score_note=score_note,
                memory=memory,
            )
            if exit_decision:
                return exit_decision
            return self._hold(ticker, "maintain short position", metadata=dict(base_metadata))

        # Entry gates
        cooldown_active = self._cooldown_active(ticker)
        if cooldown_active:
            return self._hold(ticker, "cooldown active", metadata=dict(base_metadata))
        if open_positions >= self.risk.max_positions:
            return self._hold(ticker, "max positions reached", metadata=dict(base_metadata))
        if not self.trade_tracker.can_enter():
            return self._hold(ticker, "day trade limit reached", metadata=dict(base_metadata))

        if strength >= strength_threshold:
            failing = [name for name, ok in long_gates.items() if not ok]
            if failing:
                meta = dict(base_metadata)
                meta["gate_block"] = "long:" + ",".join(failing)
                return self._hold(
                    ticker,
                    f"long gate blocked:{'/'.join(failing)}",
                    metadata=meta,
                )
            notional = self._size_trade(available_funds, per_position_cap, strength)
            if notional <= 0:
                meta = dict(base_metadata)
                meta["sizing_block"] = "notional<=0"
                return self._hold(ticker, "insufficient capital", metadata=meta)
            plan = self._calculate_exit_plan("LONG", price, strength)
            entry_metadata = {
                **base_metadata,
                "position_notional": f"{notional:.2f}",
                "target_price": f"{plan.target_price:.2f}",
                "stop_price": "none" if plan.stop_price is None else f"{plan.stop_price:.2f}",
                "exit_eta": plan.expected_exit.isoformat() if plan.expected_exit else "none",
                "tolerance": f"{plan.tolerance:.2f}",
            }
            decision = self._enter_trade(
                ticker,
                action="BUY",
                confidence=abs(strength),
                notional=notional,
                price=price,
                reason=(
                    f"long conviction prob={probability:.1%}>=min={(0.5 + strength_threshold):.1%},",
                    f" momentum={momentum:.3f}"
                ),
                metadata=entry_metadata,
            )
            if decision.quantity:
                self._remember_entry(
                    ticker,
                    side="LONG",
                    quantity=decision.quantity,
                    entry_price=price,
                    plan=plan,
                    strength=strength,
                )
            return decision

        if strength <= -strength_threshold and self.risk.allow_shorting:
            failing = [name for name, ok in short_gates.items() if not ok]
            if failing:
                meta = dict(base_metadata)
                meta["gate_block"] = "short:" + ",".join(failing)
                return self._hold(
                    ticker,
                    f"short gate blocked:{'/'.join(failing)}",
                    metadata=meta,
                )
            notional = self._size_trade(available_funds, per_position_cap, abs(strength))
            if notional <= 0:
                meta = dict(base_metadata)
                meta["sizing_block"] = "notional<=0"
                return self._hold(ticker, "insufficient capital", metadata=meta)
            plan = self._calculate_exit_plan("SHORT", price, abs(strength))
            entry_metadata = {
                **base_metadata,
                "position_notional": f"{notional:.2f}",
                "target_price": f"{plan.target_price:.2f}",
                "stop_price": "none" if plan.stop_price is None else f"{plan.stop_price:.2f}",
                "exit_eta": plan.expected_exit.isoformat() if plan.expected_exit else "none",
                "tolerance": f"{plan.tolerance:.2f}",
            }
            decision = self._enter_trade(
                ticker,
                action="SELL",
                confidence=abs(strength),
                notional=notional,
                price=price,
                reason=(
                    f"short conviction prob={(1 - probability):.1%}>=min={(0.5 + strength_threshold):.1%},",
                    f" momentum={short_momentum:.3f}"
                ),
                metadata=entry_metadata,
            )
            if decision.quantity:
                self._remember_entry(
                    ticker,
                    side="SHORT",
                    quantity=decision.quantity,
                    entry_price=price,
                    plan=plan,
                    strength=abs(strength),
                )
            return decision

        return self._hold(ticker, "signal below threshold", metadata=dict(base_metadata))

    def _evaluate_exit(
        self,
        ticker: str,
        *,
        side: str,
        quantity: float,
        price: float,
        market_value: float,
        strength: float,
        gate_snapshot: str,
        threshold: float,
        exit_threshold: float,
        score_note: Dict[str, str],
        memory: Optional[StoredPosition],
    ) -> Optional[TradeDecision]:
        record = memory or self._memory_for(ticker)
        metadata: Dict[str, str] = {**score_note, "gates": gate_snapshot}
        tolerance = self._resolve_tolerance(record)
        now = datetime.utcnow().replace(tzinfo=timezone.utc)

        if record:
            metadata.update(
                {
                    "entry_price": f"{record.entry_price:.2f}",
                    "target_price": f"{record.target_price:.2f}",
                    "stop_price": "none" if record.stop_price is None else f"{record.stop_price:.2f}",
                    "tolerance": f"{tolerance:.2f}",
                    "exit_eta": record.expected_exit.isoformat() if record.expected_exit else "none",
                }
            )

        if record and record.pending_exit:
            record.last_price = price
            record.last_updated = now
            metadata["pending_exit"] = "1"
            self._session_memory[ticker.upper()] = record
            if self.position_store:
                self.position_store.touch(ticker, price=price)
            return None

        notional = market_value if market_value > 0 else quantity * price
        trigger_reason: Optional[str] = None

        if record and record.stop_price is not None:
            if side == "LONG" and price <= record.stop_price:
                trigger_reason = f"stop triggered {price:.2f}<=plan {record.stop_price:.2f}"
            elif side == "SHORT" and price >= record.stop_price:
                trigger_reason = f"stop triggered {price:.2f}>={record.stop_price:.2f}"

        if trigger_reason is None and record:
            if side == "LONG":
                expected = max(0.0, record.target_price - record.entry_price)
                if expected > 0:
                    floor_price = record.entry_price + expected * (1 - tolerance)
                    if price >= floor_price:
                        trigger_reason = (
                            f"target met {price:.2f}>={floor_price:.2f} (goal {record.target_price:.2f})"
                        )
            else:
                expected = max(0.0, record.entry_price - record.target_price)
                if expected > 0:
                    ceiling_price = record.entry_price - expected * (1 - tolerance)
                    if price <= ceiling_price:
                        trigger_reason = (
                            f"target met {price:.2f}<={ceiling_price:.2f} (goal {record.target_price:.2f})"
                        )

        if trigger_reason is None and record and record.expected_exit and now >= record.expected_exit:
            if (side == "LONG" and price >= record.entry_price) or (
                side == "SHORT" and price <= record.entry_price
            ):
                trigger_reason = "max hold window reached"

        if trigger_reason is None:
            if side == "LONG" and strength <= exit_threshold:
                trigger_reason = "model confidence faded"
            elif side == "SHORT" and strength >= exit_threshold:
                trigger_reason = "model confidence faded"

        if trigger_reason is None:
            if record:
                record.last_price = price
                record.last_updated = now
                self._session_memory[ticker.upper()] = record
                if self.position_store:
                    self.position_store.touch(ticker, price=price)
            return None

        action = "SELL" if side == "LONG" else "BUY"
        confidence = abs(strength)
        metadata["exit_reason"] = trigger_reason
        self._mark_pending_exit(ticker, price=price, reason=trigger_reason)
        return self._exit_trade(
            ticker,
            action=action,
            confidence=confidence,
            notional=notional,
            quantity=quantity,
            reason=trigger_reason,
            metadata=metadata,
        )

    def _resolve_tolerance(self, record: Optional[StoredPosition]) -> float:
        if record is not None and record.tolerance is not None:
            value = float(record.tolerance)
        else:
            value = float(getattr(self.risk, "take_profit_tolerance", 0.1))
        return max(0.0, min(0.9, value))

    def _calculate_exit_plan(self, side: str, price: float, strength: float) -> ExitPlan:
        take_profit_pct = max(0.005, float(getattr(self.risk, "take_profit_pct", 0.08)))
        stop_loss_pct = max(0.0, float(getattr(self.risk, "stop_loss_pct", 0.04)))
        tolerance = self._resolve_tolerance(None)
        hold_minutes = int(getattr(self.risk, "expected_hold_minutes", 0) or 0)
        if hold_minutes <= 0:
            hold_minutes = max(self.risk.cooldown_minutes * 3, 720)
        expected_exit = datetime.utcnow().replace(tzinfo=timezone.utc) + timedelta(minutes=hold_minutes)

        if side == "LONG":
            target = price * (1 + take_profit_pct)
            stop = price * (1 - stop_loss_pct) if stop_loss_pct > 0 else None
        else:
            target = price * (1 - take_profit_pct)
            stop = price * (1 + stop_loss_pct) if stop_loss_pct > 0 else None

        return ExitPlan(target_price=float(target), stop_price=stop, tolerance=tolerance, expected_exit=expected_exit)

    def _remember_entry(
        self,
        ticker: str,
        *,
        side: str,
        quantity: float,
        entry_price: float,
        plan: ExitPlan,
        strength: float,
    ) -> None:
        record = StoredPosition(
            ticker=ticker.upper(),
            side=side,
            quantity=float(quantity),
            entry_price=float(entry_price),
            target_price=float(plan.target_price),
            stop_price=None if plan.stop_price is None else float(plan.stop_price),
            entry_time=datetime.utcnow().replace(tzinfo=timezone.utc),
            expected_exit=plan.expected_exit,
            tolerance=plan.tolerance,
            pending_exit=False,
            last_price=float(entry_price),
            last_updated=datetime.utcnow().replace(tzinfo=timezone.utc),
            metadata={"entry_strength": f"{strength:.3f}", "variant": "model"},
        )
        self._session_memory[ticker.upper()] = record
        if self.position_store:
            self.position_store.upsert(record)

    def _memory_for(self, ticker: str) -> Optional[StoredPosition]:
        key = ticker.upper()
        memory = self._session_memory.get(key)
        if memory is None and self.position_store:
            stored = self.position_store.get(key)
            if stored:
                self._session_memory[key] = stored
                memory = stored
        return memory

    def _sync_memory_with_broker(
        self,
        ticker: str,
        memory: StoredPosition,
        position: Dict[str, Any],
        price: float,
    ) -> None:
        key = ticker.upper()
        changed = False
        try:
            quantity = abs(float(position.get("quantity", memory.quantity)))
        except (TypeError, ValueError):
            quantity = memory.quantity
        if quantity != memory.quantity:
            memory.quantity = quantity
            changed = True
        if price > 0:
            memory.last_price = price
            changed = True
        memory.last_updated = datetime.utcnow().replace(tzinfo=timezone.utc)
        if changed:
            self._session_memory[key] = memory
            if self.position_store:
                self.position_store.upsert(memory)

    def _mark_pending_exit(self, ticker: str, *, price: float, reason: str) -> None:
        key = ticker.upper()
        record = self._session_memory.get(key)
        now = datetime.utcnow().replace(tzinfo=timezone.utc)
        if record:
            record.pending_exit = True
            record.last_price = price
            record.last_updated = now
            record.metadata["exit_request"] = reason
            self._session_memory[key] = record
        if self.position_store:
            self.position_store.mark_pending_exit(key)
            self.position_store.touch(key, price=price)

    def forget_position(self, ticker: str) -> None:
        key = ticker.upper()
        self._session_memory.pop(key, None)
        if self.position_store:
            self.position_store.remove(key)

    def trim_memory(self, active_tickers: Iterable[str]) -> None:
        active = {t.upper() for t in active_tickers}
        for ticker in list(self._session_memory.keys()):
            if ticker not in active:
                self.forget_position(ticker)

    def sync_memory_from_store(self) -> None:
        if not self.position_store:
            return
        refreshed = {record.ticker: record for record in self.position_store.all()}
        self._session_memory = refreshed

    def _parse_position(self, position: Optional[Dict[str, Any]], price: float) -> tuple[str, float, float]:
        if not position:
            return "FLAT", 0.0, 0.0
        try:
            quantity = abs(float(position.get("quantity", 0.0)))
        except (TypeError, ValueError):
            quantity = 0.0
        try:
            market_value = abs(float(position.get("market_value", quantity * price)))
        except (TypeError, ValueError):
            market_value = quantity * price
        raw_side = str(position.get("side", "")).upper()
        if quantity <= 0:
            return "FLAT", 0.0, 0.0
        if raw_side in {"LONG", "BUY"}:
            side = "LONG"
        elif raw_side in {"SHORT", "SELL"}:
            side = "SHORT"
        else:
            side = "LONG"
        return side, quantity, market_value

    def _position_budget(self, account: AccountSnapshot, market_value: float) -> tuple[float, float]:
        equity = account.equity or account.portfolio_value or account.cash
        cash = max(account.cash, 0.0)
        buying_power = max(account.buying_power, cash)
        available_funds = max(0.0, min(cash if cash > 0 else buying_power, buying_power))
        per_position_cap = (equity or buying_power or cash) * self.risk.max_capital_fraction if equity else available_funds
        if self.risk.max_position_value is not None:
            per_position_cap = min(per_position_cap, self.risk.max_position_value)
        if market_value > per_position_cap:
            per_position_cap = max(per_position_cap, market_value)
        return available_funds, per_position_cap

    def _size_trade(self, available_funds: float, per_position_cap: float, strength: float) -> float:
        if per_position_cap <= 0:
            return 0.0
        aggressiveness = max(0.4, float(getattr(self.risk, "aggressiveness", 1.0)))
        leverage_cap = max(0.25, float(getattr(self.risk, "max_trade_leverage", 1.0)))
        capital_ceiling = per_position_cap * min(leverage_cap, 2.5)
        capital_floor = max(per_position_cap, available_funds)
        base = min(capital_ceiling, capital_floor)
        if base <= 0:
            return 0.0
        conviction = max(0.0, min(1.0, strength)) ** 0.65
        ramp = (0.78 + 0.42 * min(aggressiveness, 3.5)) * conviction + 0.08
        position_fraction = max(0.06, min(leverage_cap, ramp))
        return float(base * position_fraction)

    def _cooldown_active(self, ticker: str) -> bool:
        last_trade = self.last_trade_at.get(ticker)
        if not last_trade:
            return False
        return datetime.utcnow() - last_trade < timedelta(minutes=self.risk.cooldown_minutes)

    def _hold(self, ticker: str, reason: str, metadata: Optional[Dict[str, str]] = None) -> TradeDecision:
        return TradeDecision(
            ticker=ticker,
            action="HOLD",
            confidence=0.0,
            notional=0.0,
            time_in_force="gtc",
            stop_loss=None,
            take_profit=None,
            reason=reason,
            intent="hold",
            metadata=metadata or {},
        )

    def _enter_trade(
        self,
        ticker: str,
        *,
        action: str,
        confidence: float,
        notional: float,
        price: float,
        reason: str,
        metadata: Optional[Dict[str, str]] = None,
    ) -> TradeDecision:
        if notional <= 0:
            return self._hold(ticker, "invalid notional")
        quantity = notional / price if price > 0 else None
        decision = TradeDecision(
            ticker=ticker,
            action=action,
            confidence=float(min(max(confidence, 0.0), 1.0)),
            notional=float(notional),
            time_in_force="gtc",
            stop_loss=None,
            take_profit=None,
            reason=reason,
            intent="entry",
            quantity=quantity,
            metadata=metadata or {},
        )
        self.last_trade_at[ticker] = datetime.utcnow()
        return decision

    def _exit_trade(
        self,
        ticker: str,
        *,
        action: str,
        confidence: float,
        notional: float,
        quantity: float,
        reason: str,
        metadata: Optional[Dict[str, str]] = None,
    ) -> TradeDecision:
        decision = TradeDecision(
            ticker=ticker,
            action=action,
            confidence=float(min(max(confidence, 0.0), 1.0)),
            notional=float(notional),
            time_in_force="gtc",
            stop_loss=None,
            take_profit=None,
            reason=reason,
            intent="exit",
            quantity=quantity,
            metadata=metadata or {},
        )
        self.last_trade_at[ticker] = datetime.utcnow()
        return decision


__all__ = ["ModelDecisionEngine", "SignalContext"]
