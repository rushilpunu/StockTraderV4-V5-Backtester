"""TraderV4 package initializer."""

from importlib import import_module
from typing import Any

__all__ = [
    "NewsSentimentTrader",
    "TraderConfig",
    "create_app",
    "build_trader",
    "run_server",
]


def __getattr__(name: str) -> Any:  # pragma: no cover - convenience proxy
    if name in {"NewsSentimentTrader", "TraderConfig"}:
        module = import_module("Traderv4.main")
        return getattr(module, name)
    if name in {"create_app", "build_trader"}:
        module = import_module("Traderv4.api")
        return getattr(module, name)
    if name == "run_server":
        module = import_module("Traderv4.server")
        return getattr(module, "run")
    raise AttributeError(f"module 'Traderv4' has no attribute {name!r}")
