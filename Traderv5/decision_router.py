"""Decision router that combines model predictions with cash account constraints."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Tuple

from Traderv5.bucket_scheduler import BucketScheduler, BucketType
from Traderv5.cash_ledger import CashLedger
from Traderv5.configuration import RoutingSettings

_LOG = logging.getLogger("traderv5.decision_router")


class TradeAction(Enum):
    """Possible trade actions."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    SKIP = "SKIP"


class TradeHorizon(Enum):
    """Trade horizon types."""
    SWING = "SWING"  # Multi-day hold
    INTRADAY = "INTRADAY"  # Same-day exit


@dataclass
class ModelPrediction:
    """Model prediction for a symbol."""
    symbol: str
    swing_score: float
    intraday_score: float
    confidence: float
    timestamp: datetime
    features: Dict[str, float]
    
    @property
    def best_score(self) -> float:
        """Get the best (highest) prediction score."""
        return max(self.swing_score, self.intraday_score)
    
    @property
    def best_horizon(self) -> TradeHorizon:
        """Get the horizon with the best score."""
        if self.swing_score >= self.intraday_score:
            return TradeHorizon.SWING
        else:
            return TradeHorizon.INTRADAY


@dataclass
class TradeDecision:
    """Final trade decision with all constraints applied."""
    action: TradeAction
    symbol: str
    quantity: int
    price: Decimal
    horizon: Optional[TradeHorizon]
    stop_loss: Optional[Decimal]
    take_profit: Optional[Decimal]
    risk_amount: Decimal
    reason: str
    timestamp: datetime
    model_score: float
    bucket_type: Optional[BucketType]
    
    @property
    def is_buy(self) -> bool:
        return self.action == TradeAction.BUY
    
    @property
    def is_sell(self) -> bool:
        return self.action == TradeAction.SELL
    
    @property
    def is_hold(self) -> bool:
        return self.action == TradeAction.HOLD
    
    @property
    def is_skip(self) -> bool:
        return self.action == TradeAction.SKIP


@dataclass
class PositionInfo:
    """Information about current positions."""
    symbol: str
    quantity: int
    avg_price: Decimal
    current_price: Decimal
    unrealized_pnl: Decimal
    days_held: int
    
    @property
    def market_value(self) -> Decimal:
        return Decimal(str(self.quantity)) * self.current_price


class DecisionRouter:
    """
    Routes trading decisions based on model predictions and cash account constraints.
    
    This router combines:
    1. Model predictions (swing vs intraday scores)
    2. Cash ledger constraints (settled vs unsettled cash)
    3. Bucket scheduler constraints (which bucket is active)
    4. Risk management rules
    5. Pattern day trading limits
    """
    
    def __init__(
        self,
        cash_ledger: CashLedger,
        bucket_scheduler: BucketScheduler,
        routing_settings: RoutingSettings,
    ):
        self.cash_ledger = cash_ledger
        self.bucket_scheduler = bucket_scheduler
        self.settings = routing_settings
        
        _LOG.info("Initialized decision router with settings: %s", routing_settings)
    
    def route_decision(
        self,
        prediction: ModelPrediction,
        current_positions: List[PositionInfo],
        current_price: Decimal,
        target_date: Optional[date] = None,
    ) -> TradeDecision:
        """
        Route a trading decision based on model prediction and constraints.
        
        Args:
            prediction: Model prediction for the symbol
            current_positions: Current positions (for sell decisions)
            current_price: Current market price
            target_date: Date for the decision (defaults to today)
            
        Returns:
            TradeDecision with action and details
        """
        if target_date is None:
            target_date = date.today()
        
        # Check if we have a position in this symbol
        position = self._find_position(prediction.symbol, current_positions)
        
        # Determine if we should buy, sell, or hold
        if position is not None:
            return self._route_sell_decision(prediction, position, current_price, target_date)
        else:
            return self._route_buy_decision(prediction, current_price, target_date)
    
    def _route_buy_decision(
        self,
        prediction: ModelPrediction,
        current_price: Decimal,
        target_date: date,
    ) -> TradeDecision:
        """Route a buy decision based on prediction and constraints."""
        
        # Check if any prediction meets threshold
        swing_meets_threshold = prediction.swing_score >= self.settings.swing_threshold
        intraday_meets_threshold = prediction.intraday_score >= self.settings.intraday_threshold
        
        if not swing_meets_threshold and not intraday_meets_threshold:
            return TradeDecision(
                action=TradeAction.SKIP,
                symbol=prediction.symbol,
                quantity=0,
                price=current_price,
                horizon=None,
                stop_loss=None,
                take_profit=None,
                risk_amount=Decimal('0'),
                reason=f"Model scores below threshold (swing: {prediction.swing_score:.3f}, intraday: {prediction.intraday_score:.3f})",
                timestamp=datetime.utcnow(),
                model_score=prediction.best_score,
                bucket_type=None,
            )
        
        # Determine which horizon to use
        if swing_meets_threshold and prediction.swing_score >= prediction.intraday_score:
            horizon = TradeHorizon.SWING
            score = prediction.swing_score
        elif intraday_meets_threshold:
            horizon = TradeHorizon.INTRADAY
            score = prediction.intraday_score
        else:
            horizon = TradeHorizon.SWING
            score = prediction.swing_score
        
        # Check bucket constraints
        active_bucket = self.bucket_scheduler.get_active_bucket(target_date)
        if active_bucket == BucketType.NONE:
            return TradeDecision(
                action=TradeAction.SKIP,
                symbol=prediction.symbol,
                quantity=0,
                price=current_price,
                horizon=None,
                stop_loss=None,
                take_profit=None,
                risk_amount=Decimal('0'),
                reason="No trading allowed on this day (bucket constraint)",
                timestamp=datetime.utcnow(),
                model_score=score,
                bucket_type=active_bucket,
            )
        
        # Calculate position size based on risk
        risk_amount = self._calculate_risk_amount(score, horizon)
        quantity = self._calculate_quantity(risk_amount, current_price)
        
        if quantity <= 0:
            return TradeDecision(
                action=TradeAction.SKIP,
                symbol=prediction.symbol,
                quantity=0,
                price=current_price,
                horizon=None,
                stop_loss=None,
                take_profit=None,
                risk_amount=Decimal('0'),
                reason="Position size too small after risk calculation",
                timestamp=datetime.utcnow(),
                model_score=score,
                bucket_type=active_bucket,
            )
        
        total_cost = Decimal(str(quantity)) * current_price
        
        # Check cash constraints
        can_buy, cash_reason = self.cash_ledger.can_buy(total_cost, target_date)
        if not can_buy:
            return TradeDecision(
                action=TradeAction.SKIP,
                symbol=prediction.symbol,
                quantity=0,
                price=current_price,
                horizon=None,
                stop_loss=None,
                take_profit=None,
                risk_amount=Decimal('0'),
                reason=f"Insufficient cash: {cash_reason}",
                timestamp=datetime.utcnow(),
                model_score=score,
                bucket_type=active_bucket,
            )
        
        # Check bucket constraints
        can_trade, bucket_reason = self.bucket_scheduler.can_trade(total_cost, target_date)
        if not can_trade:
            return TradeDecision(
                action=TradeAction.SKIP,
                symbol=prediction.symbol,
                quantity=0,
                price=current_price,
                horizon=None,
                stop_loss=None,
                take_profit=None,
                risk_amount=Decimal('0'),
                reason=f"Bucket constraint: {bucket_reason}",
                timestamp=datetime.utcnow(),
                model_score=score,
                bucket_type=active_bucket,
            )
        
        # Special check for intraday trades
        if horizon == TradeHorizon.INTRADAY:
            available_cash_pct = float(self.cash_ledger.state.available_cash) / float(self.cash_ledger.state.equity)
            if available_cash_pct < self.settings.intraday_min_cash_pct:
                return TradeDecision(
                    action=TradeAction.SKIP,
                    symbol=prediction.symbol,
                    quantity=0,
                    price=current_price,
                    horizon=None,
                    stop_loss=None,
                    take_profit=None,
                    risk_amount=Decimal('0'),
                    reason=f"Insufficient cash for intraday trade: {available_cash_pct:.1%} < {self.settings.intraday_min_cash_pct:.1%}",
                    timestamp=datetime.utcnow(),
                    model_score=score,
                    bucket_type=active_bucket,
                )
        
        # Calculate stop loss and take profit
        stop_loss, take_profit = self._calculate_stop_take_profit(current_price, risk_amount, quantity)
        
        return TradeDecision(
            action=TradeAction.BUY,
            symbol=prediction.symbol,
            quantity=quantity,
            price=current_price,
            horizon=horizon,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_amount=risk_amount,
            reason=f"Buy signal: {horizon.value} score {score:.3f}",
            timestamp=datetime.utcnow(),
            model_score=score,
            bucket_type=active_bucket,
        )
    
    def _route_sell_decision(
        self,
        prediction: ModelPrediction,
        position: PositionInfo,
        current_price: Decimal,
        target_date: date,
    ) -> TradeDecision:
        """Route a sell decision based on prediction and position."""
        
        # Check for sell signals (negative scores or stop loss)
        swing_sell_signal = prediction.swing_score <= -self.settings.swing_threshold
        intraday_sell_signal = prediction.intraday_score <= -self.settings.intraday_threshold
        
        # Check if we should hold due to unsettled cash needs
        if self.settings.hold_if_unsettled_needed:
            # This is a simplified check - in practice, you'd need more sophisticated logic
            # to determine if selling would create a good faith violation
            pass
        
        if swing_sell_signal or intraday_sell_signal:
            return TradeDecision(
                action=TradeAction.SELL,
                symbol=prediction.symbol,
                quantity=position.quantity,
                price=current_price,
                horizon=None,
                stop_loss=None,
                take_profit=None,
                risk_amount=Decimal('0'),
                reason=f"Sell signal: swing={prediction.swing_score:.3f}, intraday={prediction.intraday_score:.3f}",
                timestamp=datetime.utcnow(),
                model_score=prediction.best_score,
                bucket_type=None,
            )
        
        # Check for take profit or stop loss (would be calculated from position entry)
        # This is simplified - in practice, you'd track entry price and stops
        
        return TradeDecision(
            action=TradeAction.HOLD,
            symbol=prediction.symbol,
            quantity=position.quantity,
            price=current_price,
            horizon=None,
            stop_loss=None,
            take_profit=None,
            risk_amount=Decimal('0'),
            reason="No sell signal, holding position",
            timestamp=datetime.utcnow(),
            model_score=prediction.best_score,
            bucket_type=None,
        )
    
    def _find_position(self, symbol: str, positions: List[PositionInfo]) -> Optional[PositionInfo]:
        """Find position for a given symbol."""
        for position in positions:
            if position.symbol == symbol:
                return position
        return None
    
    def _calculate_risk_amount(self, score: float, horizon: TradeHorizon) -> Decimal:
        """Calculate risk amount based on model score and horizon."""
        # Base risk from cash ledger settings
        base_risk_pct = Decimal('0.02')  # 2% default
        
        # Adjust based on score confidence
        confidence_multiplier = Decimal(str(min(2.0, max(0.5, score))))
        
        # Adjust based on horizon
        if horizon == TradeHorizon.INTRADAY:
            horizon_multiplier = Decimal('0.5')  # Lower risk for intraday
        else:
            horizon_multiplier = Decimal('1.0')
        
        # Get current equity
        equity = self.cash_ledger.state.equity
        
        # Calculate risk amount
        risk_amount = equity * base_risk_pct * confidence_multiplier * horizon_multiplier
        
        return risk_amount.quantize(Decimal('0.01'))
    
    def _calculate_quantity(self, risk_amount: Decimal, price: Decimal) -> int:
        """Calculate position quantity based on risk amount and price."""
        if price <= 0:
            return 0
        
        # Simple calculation - in practice, you'd use ATR for stop loss distance
        stop_distance_pct = Decimal('0.05')  # 5% stop loss
        stop_distance = price * stop_distance_pct
        
        if stop_distance <= 0:
            return 0
        
        quantity = risk_amount / stop_distance
        # Ensure at least one share if we can afford it
        q_int = int(quantity)
        if q_int <= 0 and risk_amount >= price:
            q_int = 1
        return max(0, q_int)
    
    def _calculate_stop_take_profit(
        self,
        entry_price: Decimal,
        risk_amount: Decimal,
        quantity: int,
    ) -> Tuple[Decimal, Decimal]:
        """Calculate stop loss and take profit levels."""
        # Simple calculation - in practice, you'd use ATR
        stop_distance_pct = Decimal('0.05')  # 5% stop loss
        take_profit_multiple = Decimal('2.5')  # 2.5x risk
        
        stop_loss = entry_price * (Decimal('1') - stop_distance_pct)
        take_profit = entry_price + (entry_price * stop_distance_pct * take_profit_multiple)
        
        return stop_loss.quantize(Decimal('0.01')), take_profit.quantize(Decimal('0.01'))


__all__ = [
    "DecisionRouter",
    "TradeAction",
    "TradeHorizon",
    "ModelPrediction",
    "TradeDecision",
    "PositionInfo",
]
