"""Shared risk profile presets for TraderV5."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, MutableMapping

from Traderv4.funcs import RiskConfig


@dataclass(frozen=True)
class RiskProfile:
    """Represents a named risk posture that can be reused across entry points."""

    name: str
    entry_threshold: float
    exit_threshold: float
    aggressiveness: float
    entry_bias: float
    leverage: float
    cooldown: int
    max_positions: int
    max_capital_fraction: float
    description: str


_PROFILES: Dict[str, RiskProfile] = {
    "baseline": RiskProfile(
        name="baseline",
        entry_threshold=0.03,
        exit_threshold=0.012,
        aggressiveness=1.5,
        entry_bias=0.0,
        leverage=1.5,
        cooldown=4,
        max_positions=3,
        max_capital_fraction=0.29,
        description="Conservative sizing tuned for steadier equity growth.",
    ),
    "balanced": RiskProfile(
        name="balanced",
        entry_threshold=0.024,
        exit_threshold=0.01,
        aggressiveness=2.0,
        entry_bias=0.02,
        leverage=2.1,
        cooldown=3,
        max_positions=4,
        max_capital_fraction=0.35,
        description="Balanced exposure with dynamic conviction scaling.",
    ),
    "aggressive": RiskProfile(
        name="aggressive",
        entry_threshold=0.022,
        exit_threshold=0.007,
        aggressiveness=2.6,
        entry_bias=0.04,
        leverage=1.4,
        cooldown=2,
        max_positions=5,
        max_capital_fraction=0.195,
        description="High-conviction profile that leans into strong sentiment runs.",
    ),
    "swing": RiskProfile(
        name="swing",
        entry_threshold=0.052,
        exit_threshold=0.02,
        aggressiveness=1.25,
        entry_bias=0.01,
        leverage=1.0,
        cooldown=720,
        max_positions=3,
        max_capital_fraction=0.38,
        description=(
            "Overnight swing posture tuned for $10k accounts and pattern day trading limits."
        ),
    ),
}

AggressiveProfile: RiskProfile = _PROFILES["aggressive"]
SwingProfile: RiskProfile = _PROFILES["swing"]


def get_profile(name: str) -> RiskProfile:
    try:
        return _PROFILES[name.lower()]
    except KeyError as exc:  # pragma: no cover - defensive guard
        valid = ", ".join(sorted(_PROFILES))
        raise KeyError(f"Unknown TraderV5 risk profile '{name}'. Valid options: {valid}") from exc


def apply_profile(
    risk: RiskConfig,
    profile: RiskProfile,
    overrides: Mapping[str, float] | None = None,
) -> RiskConfig:
    """Apply a profile onto an existing ``RiskConfig`` instance in-place."""

    values: MutableMapping[str, float] = dict(overrides or {})
    risk.max_capital_fraction = float(values.pop("max_capital_fraction", profile.max_capital_fraction))
    risk.max_positions = int(values.pop("max_positions", profile.max_positions))
    risk.cooldown_minutes = int(values.pop("cooldown", profile.cooldown))
    risk.entry_sentiment_threshold = float(values.pop("entry", profile.entry_threshold))
    risk.exit_sentiment_threshold = float(values.pop("exit", profile.exit_threshold))
    risk.aggressiveness = float(values.pop("aggressiveness", profile.aggressiveness))
    risk.entry_signal_bias = float(values.pop("bias", profile.entry_bias))
    risk.max_trade_leverage = float(values.pop("leverage", profile.leverage))
    if "allow_shorting" in values:
        risk.allow_shorting = bool(values.pop("allow_shorting"))
    return risk


__all__ = [
    "RiskProfile",
    "AggressiveProfile",
    "SwingProfile",
    "apply_profile",
    "get_profile",
]
