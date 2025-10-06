"""Risk management system with ATR-based stops and position sizing."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from Traderv5.configuration import RiskSettings

_LOG = logging.getLogger("traderv5.risk_manager")


@dataclass
class RiskMetrics:
    """Risk metrics for a position or trade."""
    symbol: str
    entry_price: Decimal
    current_price: Decimal
    quantity: int
    atr: Decimal
    atr_multiplier: Decimal
    stop_loss: Decimal
    take_profit: Decimal
    risk_amount: Decimal
    reward_risk_ratio: Decimal
    unrealized_pnl: Decimal
    unrealized_pnl_pct: Decimal
    
    @property
    def market_value(self) -> Decimal:
        return Decimal(str(self.quantity)) * self.current_price
    
    @property
    def is_stopped_out(self) -> bool:
        return self.current_price <= self.stop_loss
    
    @property
    def is_take_profit(self) -> bool:
        return self.current_price >= self.take_profit


@dataclass
class AccountRiskState:
    """Current risk state of the account."""
    total_equity: Decimal
    available_cash: Decimal
    total_risk: Decimal
    max_drawdown: Decimal
    current_drawdown: Decimal
    risk_per_trade_pct: Decimal
    position_count: int
    last_updated: datetime
    
    @property
    def risk_utilization_pct(self) -> Decimal:
        """Percentage of equity at risk."""
        if self.total_equity <= 0:
            return Decimal('0')
        return (self.total_risk / self.total_equity) * Decimal('100')
    
    @property
    def drawdown_pct(self) -> Decimal:
        """Current drawdown percentage."""
        if self.max_drawdown <= 0:
            return Decimal('0')
        return (self.current_drawdown / self.max_drawdown) * Decimal('100')


class RiskManager:
    """
    Manages risk for trading positions using ATR-based stops and position sizing.
    
    Features:
    - ATR-based stop loss calculation
    - Position sizing based on risk percentage
    - Drawdown-based position size reduction
    - Risk limit enforcement
    - Take profit calculation
    """
    
    def __init__(self, risk_settings: RiskSettings):
        self.settings = risk_settings
        self.risk_state = AccountRiskState(
            total_equity=Decimal('0'),
            available_cash=Decimal('0'),
            total_risk=Decimal('0'),
            max_drawdown=Decimal('0'),
            current_drawdown=Decimal('0'),
            risk_per_trade_pct=Decimal(str(risk_settings.risk_per_trade_pct)),
            position_count=0,
            last_updated=datetime.utcnow(),
        )
        
        _LOG.info("Initialized risk manager with settings: %s", risk_settings)
    
    def update_account_state(
        self,
        total_equity: Decimal,
        available_cash: Decimal,
        current_positions: List[RiskMetrics],
    ) -> None:
        """Update the account risk state."""
        self.risk_state.total_equity = total_equity
        self.risk_state.available_cash = available_cash
        self.risk_state.position_count = len(current_positions)
        
        # Calculate total risk from current positions
        total_risk = Decimal('0')
        for position in current_positions:
            risk_per_share = position.entry_price - position.stop_loss
            position_risk = risk_per_share * Decimal(str(position.quantity))
            total_risk += position_risk
        
        self.risk_state.total_risk = total_risk
        self.risk_state.last_updated = datetime.utcnow()
        
        _LOG.debug(
            "Updated risk state: equity=$%.2f, risk=$%.2f (%.1f%%), positions=%d",
            total_equity, total_risk, self.risk_state.risk_utilization_pct, len(current_positions)
        )
    
    def calculate_atr(self, price_data: pd.DataFrame, period: int = 14) -> Decimal:
        """
        Calculate Average True Range (ATR) from price data.
        
        Args:
            price_data: DataFrame with OHLC data
            period: ATR period (default 14)
            
        Returns:
            ATR value
        """
        if len(price_data) < period + 1:
            _LOG.warning("Insufficient data for ATR calculation: need %d, have %d", period + 1, len(price_data))
            return Decimal('0')
        
        # Calculate True Range
        high = price_data['high'].values
        low = price_data['low'].values
        close = price_data['close'].values
        
        # Shift close prices for previous day
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]  # First value uses current close
        
        # Calculate True Range components
        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        
        # True Range is the maximum of the three
        true_range = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Calculate ATR as simple moving average of True Range
        atr = np.mean(true_range[-period:])
        
        return Decimal(str(atr)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
    
    def calculate_stop_loss(
        self,
        entry_price: Decimal,
        atr: Decimal,
        is_long: bool = True,
    ) -> Decimal:
        """
        Calculate stop loss based on ATR.
        
        Args:
            entry_price: Entry price of the position
            atr: Average True Range
            is_long: True for long positions, False for short
            
        Returns:
            Stop loss price
        """
        if atr <= 0:
            # Fallback to percentage-based stop if ATR is invalid
            stop_distance_pct = Decimal('0.05')  # 5%
            stop_distance = entry_price * stop_distance_pct
        else:
            stop_distance = atr * Decimal(str(self.settings.atr_stop_multiplier))
        
        if is_long:
            stop_loss = entry_price - stop_distance
        else:
            stop_loss = entry_price + stop_distance
        
        return stop_loss.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
    
    def calculate_take_profit(
        self,
        entry_price: Decimal,
        stop_loss: Decimal,
        is_long: bool = True,
    ) -> Decimal:
        """
        Calculate take profit based on risk-reward ratio.
        
        Args:
            entry_price: Entry price of the position
            stop_loss: Stop loss price
            is_long: True for long positions, False for short
            
        Returns:
            Take profit price
        """
        risk_per_share = abs(entry_price - stop_loss)
        reward_per_share = risk_per_share * Decimal(str(self.settings.take_profit_multiple))
        
        if is_long:
            take_profit = entry_price + reward_per_share
        else:
            take_profit = entry_price - reward_per_share
        
        return take_profit.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
    
    def calculate_position_size(
        self,
        entry_price: Decimal,
        stop_loss: Decimal,
        available_cash: Decimal,
        max_risk_amount: Optional[Decimal] = None,
    ) -> Tuple[int, Decimal]:
        """
        Calculate position size based on risk management rules.
        
        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            available_cash: Available cash for trading
            max_risk_amount: Maximum risk amount (defaults to risk_per_trade_pct)
            
        Returns:
            Tuple of (quantity, actual_risk_amount)
        """
        if entry_price <= 0 or stop_loss <= 0:
            return 0, Decimal('0')
        
        # Calculate risk per share
        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share <= 0:
            return 0, Decimal('0')
        
        # Determine maximum risk amount
        if max_risk_amount is None:
            max_risk_amount = self.risk_state.total_equity * self.risk_state.risk_per_trade_pct
        
        # Apply drawdown-based size reduction
        if self.risk_state.current_drawdown > 0:
            drawdown_pct = self.risk_state.current_drawdown / self.risk_state.max_drawdown
            if drawdown_pct > Decimal(str(self.settings.max_drawdown_pct)):
                size_reduction = Decimal(str(self.settings.drawdown_size_reduction))
                max_risk_amount *= size_reduction
                _LOG.info(
                    "Reducing position size due to drawdown: %.1f%% > %.1f%%, reduction factor: %.2f",
                    drawdown_pct * 100, self.settings.max_drawdown_pct * 100, size_reduction
                )
        
        # Calculate quantity based on risk
        max_quantity_by_risk = int(max_risk_amount / risk_per_share)
        
        # Calculate quantity based on available cash
        max_quantity_by_cash = int(available_cash / entry_price)
        
        # Use the smaller of the two
        quantity = min(max_quantity_by_risk, max_quantity_by_cash)
        
        # Ensure minimum quantity
        if quantity < 1:
            return 0, Decimal('0')
        
        # Calculate actual risk amount
        actual_risk_amount = risk_per_share * Decimal(str(quantity))
        
        return quantity, actual_risk_amount
    
    def create_risk_metrics(
        self,
        symbol: str,
        entry_price: Decimal,
        quantity: int,
        atr: Decimal,
        current_price: Optional[Decimal] = None,
    ) -> RiskMetrics:
        """
        Create risk metrics for a position.
        
        Args:
            symbol: Stock symbol
            entry_price: Entry price
            quantity: Position quantity
            atr: Average True Range
            current_price: Current market price (defaults to entry price)
            
        Returns:
            RiskMetrics object
        """
        if current_price is None:
            current_price = entry_price
        
        # Calculate stop loss and take profit
        stop_loss = self.calculate_stop_loss(entry_price, atr, is_long=True)
        take_profit = self.calculate_take_profit(entry_price, stop_loss, is_long=True)
        
        # Calculate risk and reward
        risk_per_share = entry_price - stop_loss
        reward_per_share = take_profit - entry_price
        risk_amount = risk_per_share * Decimal(str(quantity))
        reward_risk_ratio = reward_per_share / risk_per_share if risk_per_share > 0 else Decimal('0')
        
        # Calculate unrealized P&L
        unrealized_pnl = (current_price - entry_price) * Decimal(str(quantity))
        unrealized_pnl_pct = (current_price - entry_price) / entry_price * Decimal('100')
        
        return RiskMetrics(
            symbol=symbol,
            entry_price=entry_price,
            current_price=current_price,
            quantity=quantity,
            atr=atr,
            atr_multiplier=Decimal(str(self.settings.atr_stop_multiplier)),
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_amount=risk_amount,
            reward_risk_ratio=reward_risk_ratio,
            unrealized_pnl=unrealized_pnl,
            unrealized_pnl_pct=unrealized_pnl_pct,
        )
    
    def check_risk_limits(
        self,
        new_position_risk: Decimal,
        current_positions: List[RiskMetrics],
    ) -> Tuple[bool, str]:
        """
        Check if adding a new position would violate risk limits.
        
        Args:
            new_position_risk: Risk amount of the new position
            current_positions: Current positions
            
        Returns:
            Tuple of (allowed, reason)
        """
        # Calculate total risk including new position
        current_total_risk = sum(pos.risk_amount for pos in current_positions)
        new_total_risk = current_total_risk + new_position_risk
        
        # Check maximum risk per trade
        max_risk_per_trade = self.risk_state.total_equity * self.risk_state.risk_per_trade_pct
        if new_position_risk > max_risk_per_trade:
            return False, f"Position risk ${new_position_risk} exceeds maximum ${max_risk_per_trade}"
        
        # Check total portfolio risk (e.g., max 10% of equity at risk)
        max_portfolio_risk = self.risk_state.total_equity * Decimal('0.10')
        if new_total_risk > max_portfolio_risk:
            return False, f"Total portfolio risk ${new_total_risk} would exceed maximum ${max_portfolio_risk}"
        
        # Check position count limit
        max_positions = 10  # Configurable limit
        if len(current_positions) >= max_positions:
            return False, f"Maximum positions ({max_positions}) already reached"
        
        return True, "Risk limits satisfied"
    
    def update_drawdown(self, peak_equity: Decimal, current_equity: Decimal) -> None:
        """Update drawdown tracking."""
        if current_equity > peak_equity:
            # New peak
            self.risk_state.max_drawdown = peak_equity
            self.risk_state.current_drawdown = Decimal('0')
        else:
            # Calculate current drawdown
            self.risk_state.current_drawdown = peak_equity - current_equity
        
        _LOG.debug(
            "Drawdown update: peak=$%.2f, current=$%.2f, drawdown=$%.2f (%.1f%%)",
            peak_equity, current_equity, self.risk_state.current_drawdown,
            self.risk_state.drawdown_pct
        )
    
    def get_risk_summary(self) -> Dict[str, float]:
        """Get a summary of current risk metrics."""
        return {
            "total_equity": float(self.risk_state.total_equity),
            "available_cash": float(self.risk_state.available_cash),
            "total_risk": float(self.risk_state.total_risk),
            "risk_utilization_pct": float(self.risk_state.risk_utilization_pct),
            "current_drawdown": float(self.risk_state.current_drawdown),
            "drawdown_pct": float(self.risk_state.drawdown_pct),
            "position_count": self.risk_state.position_count,
            "risk_per_trade_pct": float(self.risk_state.risk_per_trade_pct),
        }


__all__ = [
    "RiskManager",
    "RiskMetrics",
    "AccountRiskState",
]
