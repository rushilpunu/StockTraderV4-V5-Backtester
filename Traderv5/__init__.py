"""Trader V5 package with ML-driven decision making."""

from importlib import import_module
from typing import Any

__all__ = ["ModelDrivenTrader", "ModelDecisionEngine"]


def __getattr__(name: str) -> Any:  # pragma: no cover - convenience proxy
    if name == "ModelDrivenTrader":
        module = import_module("Traderv5.trader")
        return module.ModelDrivenTrader
    if name == "ModelDecisionEngine":
        module = import_module("Traderv5.decision")
        return module.ModelDecisionEngine
    raise AttributeError(f"module 'Traderv5' has no attribute {name!r}")
