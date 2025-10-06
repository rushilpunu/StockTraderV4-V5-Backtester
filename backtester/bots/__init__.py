"""Available bot adapters."""

from .base import BOT_REGISTRY, BaseBot
from .traderv4_bot import TraderV4Bot
from Traderv5.backtest_adapter import TraderV5Bot  # noqa: F401

__all__ = ["BOT_REGISTRY", "BaseBot", "TraderV4Bot", "TraderV5Bot"]
