"""Metric helpers for backtesting results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from .simulator import ExecutedTrade


@dataclass
class BacktestMetrics:
    total_return: float
    trades: List[ExecutedTrade]
    pnl_curve: List[float]
    profitable_trades: int
    total_alerts: int
    debug: Dict[str, Any] = field(default_factory=dict)
    feature_summary: Dict[str, Any] = field(default_factory=dict)

    @property
    def alert_precision(self) -> float:
        if self.total_alerts == 0:
            return 0.0
        return self.profitable_trades / self.total_alerts

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        return self.profitable_trades / len(self.trades)

    @property
    def max_drawdown(self) -> float:
        if not self.pnl_curve:
            return 0.0
        peak = self.pnl_curve[0]
        max_dd = 0.0
        for value in self.pnl_curve:
            peak = max(peak, value)
            dd = (peak - value) / peak if peak else 0.0
            if dd > max_dd:
                max_dd = dd
        return max_dd


__all__ = ["BacktestMetrics"]
