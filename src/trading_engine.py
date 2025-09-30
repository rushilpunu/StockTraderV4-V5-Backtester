"""Trading decision engine with comprehensive risk management."""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import statistics
from loguru import logger

# Import config from the small capital trader directory
import sys
import os
small_capital_path = os.path.join(os.path.dirname(__file__), '..', 'small_capital_trader')
sys.path.insert(0, small_capital_path)
from config import SmallCapitalTradingConfig
config = SmallCapitalTradingConfig()
from alert_system import TradingAlert, AlertLevel
from volatility_analyzer import VolatilitySignal
from data_processor import EventData


class TradeAction(Enum):
    """Trading actions."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"


class OrderType(Enum):
    """Order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop"
    TAKE_PROFIT = "limit"


@dataclass
class TradingDecision:
    """Container for trading decisions."""
    ticker: str
    action: TradeAction
    quantity: int
    order_type: OrderType
    price: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    
    # Decision metadata
    confidence: float
    reasoning: str
    risk_score: float
    expected_return: float
    max_risk: float
    
    # Source information
    alert_id: Optional[str]
    signal_strength: float
    sentiment_score: float
    
    # Timestamps
    decision_time: datetime
    valid_until: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'ticker': self.ticker,
            'action': self.action.value,
            'quantity': self.quantity,
            'order_type': self.order_type.value,
            'price': self.price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'confidence': self.confidence,
            'reasoning': self.reasoning,
            'risk_score': self.risk_score,
            'expected_return': self.expected_return,
            'max_risk': self.max_risk,
            'alert_id': self.alert_id,
            'signal_strength': self.signal_strength,
            'sentiment_score': self.sentiment_score,
            'decision_time': self.decision_time.isoformat(),
            'valid_until': self.valid_until.isoformat()
        }


@dataclass
class Position:
    """Container for position information."""
    ticker: str
    quantity: int
    entry_price: float
    current_price: float
    entry_time: datetime
    stop_loss: Optional[float]
    take_profit: Optional[float]
    
    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price
    
    @property
    def unrealized_pnl(self) -> float:
        return (self.current_price - self.entry_price) * self.quantity
    
    @property
    def unrealized_pnl_percent(self) -> float:
        return (self.current_price - self.entry_price) / self.entry_price * 100


@dataclass
class RiskMetrics:
    """Container for risk assessment metrics."""
    portfolio_value: float
    total_exposure: float
    max_position_risk: float
    portfolio_beta: float
    var_95: float  # Value at Risk 95%
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    risk_adjusted_return: float


class TradingEngine:
    """Advanced trading decision engine with risk management."""
    
    def __init__(self):
        self.positions: Dict[str, Position] = {}
        self.pending_orders: List[TradingDecision] = []
        self.trade_history: List[Dict[str, Any]] = []
        self.portfolio_value: float = 10000.0  # Starting portfolio value
        self.max_portfolio_risk: float = 0.02  # Max 2% portfolio risk per trade
        self.max_sector_exposure: float = 0.3  # Max 30% in any sector
        self.max_single_position: float = 0.1  # Max 10% in single position
        
        # Risk management state
        self.daily_loss_limit: float = 0.05  # Max 5% daily loss
        self.daily_pnl: float = 0.0
        self.consecutive_losses: int = 0
        self.last_trade_time: Dict[str, datetime] = {}
        
        # Performance tracking
        self.total_trades: int = 0
        self.winning_trades: int = 0
        self.total_pnl: float = 0.0
        
    async def evaluate_trading_decision(
        self, 
        alert: TradingAlert,
        current_price: float,
        market_data: Dict[str, Any] = None
    ) -> Optional[TradingDecision]:
        """
        Evaluate whether to execute a trade based on an alert.
        
        Args:
            alert: Trading alert to evaluate
            current_price: Current stock price
            market_data: Additional market data (optional)
            
        Returns:
            TradingDecision if trade should be executed, None otherwise
        """
        ticker = alert.ticker
        
        # Pre-flight risk checks
        if not self._pre_flight_risk_check(ticker, alert):
            logger.info(f"Pre-flight risk check failed for {ticker}")
            return None
        
        # Analyze the alert and determine action
        action = self._determine_trading_action(alert, current_price)
        
        if action == TradeAction.HOLD:
            logger.info(f"Decision: HOLD for {ticker}")
            return None
        
        # Calculate position size
        position_size = self._calculate_position_size(alert, current_price)
        
        if position_size <= 0:
            logger.info(f"Position size calculation returned 0 for {ticker}")
            return None
        
        # Calculate risk management levels
        stop_loss, take_profit = self._calculate_risk_levels(
            action, current_price, alert
        )
        
        # Determine order type
        order_type = self._determine_order_type(alert, current_price)
        
        # Calculate expected metrics
        risk_score = self._calculate_trade_risk_score(
            ticker, action, position_size, current_price, stop_loss
        )
        
        expected_return = self._calculate_expected_return(
            action, current_price, take_profit, alert.confidence
        )
        
        # Final risk assessment
        if risk_score > 0.9:  # Make threshold more permissive
            logger.warning(f"Trade risk too high for {ticker}: {risk_score}")
            return None
        
        # Generate reasoning
        reasoning = self._generate_trading_reasoning(alert, action, risk_score)
        
        decision = TradingDecision(
            ticker=ticker,
            action=action,
            quantity=position_size,
            order_type=order_type,
            price=current_price if order_type == OrderType.MARKET else None,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=alert.confidence,
            reasoning=reasoning,
            risk_score=risk_score,
            expected_return=expected_return,
            max_risk=position_size * current_price * config.stop_loss_percentage,
            alert_id=alert.alert_id,
            signal_strength=alert.volatility_score,
            sentiment_score=alert.sentiment_score,
            decision_time=datetime.utcnow(),
            valid_until=datetime.utcnow() + timedelta(minutes=15)
        )
        
        logger.info(
            f"Trading decision: {action.value.upper()} {position_size} shares of {ticker} "
            f"(confidence: {alert.confidence:.1%}, risk: {risk_score:.3f})"
        )
        
        return decision
    
    def _pre_flight_risk_check(self, ticker: str, alert: TradingAlert) -> bool:
        """Perform pre-flight risk checks before considering a trade."""
        
        # Check daily loss limit
        if self.daily_pnl < -self.daily_loss_limit * self.portfolio_value:
            logger.warning("Daily loss limit reached")
            return False
        
        # Check consecutive losses
        if self.consecutive_losses >= 3:
            logger.warning("Too many consecutive losses")
            return False
        
        # Check cooldown period
        if ticker in self.last_trade_time:
            time_since_last = datetime.utcnow() - self.last_trade_time[ticker]
            if time_since_last < timedelta(minutes=config.cooldown_minutes):
                logger.info(f"Cooldown period active for {ticker}")
                return False
        
        # Check alert confidence and level
        if alert.confidence < 0.25:
            logger.info(f"Alert confidence too low: {alert.confidence}")
            return False
        
        if alert.alert_level == AlertLevel.LOW and alert.confidence < 0.5:
            logger.info("Alert level too low for trading")
            return False
        
        # Check if we already have a position in this ticker
        if ticker in self.positions:
            position = self.positions[ticker]
            # Don't add to position if it's already at max size
            position_value = abs(position.market_value)
            max_position_value = self.portfolio_value * self.max_single_position
            
            if position_value >= max_position_value:
                logger.info(f"Position in {ticker} already at maximum size")
                return False
        
        return True
    
    def _determine_trading_action(
        self, 
        alert: TradingAlert, 
        current_price: float
    ) -> TradeAction:
        """Determine the appropriate trading action based on alert."""
        
        recommended_action = alert.recommended_action.lower()
        
        # Check if we have an existing position
        if alert.ticker in self.positions:
            position = self.positions[alert.ticker]
            
            # If we have a long position and get a sell signal
            if position.quantity > 0 and recommended_action == 'sell':
                return TradeAction.SELL
            
            # If we have a short position and get a buy signal
            elif position.quantity < 0 and recommended_action == 'buy':
                return TradeAction.BUY
            
            # If we have a position and get a conflicting signal, consider closing
            elif ((position.quantity > 0 and recommended_action == 'sell') or
                  (position.quantity < 0 and recommended_action == 'buy')):
                return TradeAction.CLOSE
        
        # No existing position, follow the recommendation
        if recommended_action == 'buy':
            return TradeAction.BUY
        elif recommended_action == 'sell':
            return TradeAction.SELL
        else:
            return TradeAction.HOLD
    
    def _calculate_position_size(
        self, 
        alert: TradingAlert, 
        current_price: float
    ) -> int:
        """Calculate appropriate position size based on risk management."""
        
        # Start with recommended position size from alert
        base_size = alert.position_size_recommendation
        
        # Adjust based on portfolio risk limits
        max_risk_amount = self.portfolio_value * self.max_portfolio_risk
        
        # Calculate risk per share (distance to stop loss)
        if alert.stop_loss_suggestion:
            risk_per_share = abs(current_price - (current_price * (1 - alert.stop_loss_suggestion)))
            max_shares_by_risk = int(max_risk_amount / risk_per_share) if risk_per_share > 0 else 0
        else:
            # Use default stop loss
            risk_per_share = current_price * config.stop_loss_percentage
            max_shares_by_risk = int(max_risk_amount / risk_per_share)
        
        # Adjust based on position size limits
        max_position_value = self.portfolio_value * self.max_single_position
        max_shares_by_position = int(max_position_value / current_price)
        
        # Adjust based on confidence
        confidence_multiplier = min(alert.confidence, 1.0)
        
        # Take the minimum of all constraints
        position_size = min(
            base_size,
            max_shares_by_risk,
            max_shares_by_position
        )
        
        # Apply confidence multiplier
        position_size = int(position_size * confidence_multiplier)
        
        # Ensure minimum viable position
        min_position_value = 50  # Minimum $50 position (more aggressive)
        min_shares = max(1, int(min_position_value / current_price))
        
        return max(min_shares, position_size) if position_size > 0 else 0
    
    def _calculate_risk_levels(
        self, 
        action: TradeAction, 
        current_price: float, 
        alert: TradingAlert
    ) -> Tuple[Optional[float], Optional[float]]:
        """Calculate stop loss and take profit levels."""
        
        if action == TradeAction.HOLD:
            return None, None
        
        # Get suggested levels from alert
        stop_loss_pct = alert.stop_loss_suggestion or config.stop_loss_percentage
        take_profit_pct = alert.take_profit_suggestion or config.take_profit_percentage
        
        # Adjust based on volatility
        volatility_multiplier = 1.0
        if alert.volatility_score > 0.8:
            volatility_multiplier = 1.3  # Wider stops for high volatility
        elif alert.volatility_score > 0.6:
            volatility_multiplier = 1.15
        
        stop_loss_pct *= volatility_multiplier
        take_profit_pct *= volatility_multiplier
        
        # Calculate actual levels based on action
        if action in [TradeAction.BUY, TradeAction.CLOSE]:
            stop_loss = current_price * (1 - stop_loss_pct)
            take_profit = current_price * (1 + take_profit_pct)
        else:  # SELL
            stop_loss = current_price * (1 + stop_loss_pct)
            take_profit = current_price * (1 - take_profit_pct)
        
        return stop_loss, take_profit
    
    def _determine_order_type(self, alert: TradingAlert, current_price: float) -> OrderType:
        """Determine the appropriate order type."""
        
        # Use market orders for higher-confidence, urgent signals (more aggressive)
        if (alert.alert_level in [AlertLevel.MEDIUM, AlertLevel.HIGH, AlertLevel.CRITICAL] and
            alert.confidence > 0.55):
            return OrderType.MARKET
        
        # Allow market orders for LOW alerts with sufficient confidence
        if (alert.alert_level == AlertLevel.LOW and alert.confidence > 0.6):
            return OrderType.MARKET

        # Use limit orders for everything else
        return OrderType.LIMIT
    
    def _calculate_trade_risk_score(
        self,
        ticker: str,
        action: TradeAction,
        position_size: int,
        current_price: float,
        stop_loss: Optional[float]
    ) -> float:
        """Calculate comprehensive risk score for the trade."""
        
        risk_factors = []
        
        # Position size risk
        position_value = position_size * current_price
        position_risk = position_value / self.portfolio_value
        risk_factors.append(min(position_risk / self.max_single_position, 1.0))
        
        # Stop loss risk
        if stop_loss:
            stop_loss_risk = abs(current_price - stop_loss) / current_price
            risk_factors.append(min(stop_loss_risk / 0.1, 1.0))  # Normalize to 10% max
        else:
            risk_factors.append(0.8)  # High risk if no stop loss
        
        # Portfolio concentration risk
        current_exposure = sum(abs(pos.market_value) for pos in self.positions.values())
        total_exposure = current_exposure + position_value
        concentration_risk = total_exposure / self.portfolio_value
        risk_factors.append(min(concentration_risk, 1.0))
        
        # Recent performance risk
        if self.consecutive_losses > 0:
            performance_risk = self.consecutive_losses / 5.0  # Max 5 losses
            risk_factors.append(min(performance_risk, 1.0))
        else:
            risk_factors.append(0.0)
        
        # Daily PnL risk
        if self.daily_pnl < 0:
            daily_risk = abs(self.daily_pnl) / (self.daily_loss_limit * self.portfolio_value)
            risk_factors.append(min(daily_risk, 1.0))
        else:
            risk_factors.append(0.0)
        
        return statistics.mean(risk_factors)
    
    def _calculate_expected_return(
        self,
        action: TradeAction,
        current_price: float,
        take_profit: Optional[float],
        confidence: float
    ) -> float:
        """Calculate expected return for the trade."""
        
        if not take_profit or action == TradeAction.HOLD:
            return 0.0
        
        # Calculate potential return
        if action in [TradeAction.BUY, TradeAction.CLOSE]:
            potential_return = (take_profit - current_price) / current_price
        else:  # SELL
            potential_return = (current_price - take_profit) / current_price
        
        # Adjust by confidence (probability of success)
        expected_return = potential_return * confidence
        
        return expected_return
    
    def _generate_trading_reasoning(
        self,
        alert: TradingAlert,
        action: TradeAction,
        risk_score: float
    ) -> str:
        """Generate human-readable reasoning for the trading decision."""
        
        reasoning_parts = [
            f"{action.value.title()} signal based on {alert.alert_type.value}",
            f"Alert confidence: {alert.confidence:.1%}",
            f"Sentiment: {alert.sentiment_score:.3f}",
            f"Volatility: {alert.volatility_score:.3f}",
            f"Risk score: {risk_score:.3f}"
        ]
        
        if alert.key_themes:
            reasoning_parts.append(f"Key themes: {', '.join(alert.key_themes[:2])}")
        
        if alert.top_headlines:
            reasoning_parts.append(f"Based on {len(alert.top_headlines)} recent headlines")
        
        return "; ".join(reasoning_parts)
    
    def update_position(
        self,
        ticker: str,
        quantity: int,
        price: float,
        action: TradeAction,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ):
        """Update position after trade execution."""
        
        if action == TradeAction.CLOSE:
            # Full close requested
            if ticker in self.positions:
                position = self.positions[ticker]
                pnl = (price - position.entry_price) * position.quantity
                self._record_trade_pnl(pnl)
                del self.positions[ticker]
            self.last_trade_time[ticker] = datetime.utcnow()
            logger.info(f"Position closed for {ticker}")
            return
        
        # Determine signed delta quantity based on action
        delta_quantity = quantity if action == TradeAction.BUY else -quantity
        
        if ticker not in self.positions:
            if delta_quantity != 0:
                self.positions[ticker] = Position(
                    ticker=ticker,
                    quantity=delta_quantity,
                    entry_price=price,
                    current_price=price,
                    entry_time=datetime.utcnow(),
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )
        else:
            position = self.positions[ticker]
            
            # Update existing position
            total_quantity = position.quantity + delta_quantity
            if total_quantity == 0:
                # Position closed
                pnl = (price - position.entry_price) * position.quantity
                self._record_trade_pnl(pnl)
                del self.positions[ticker]
            else:
                # Update position
                weighted_price = (
                    (position.quantity * position.entry_price + delta_quantity * price) /
                    total_quantity
                )
                position.quantity = total_quantity
                position.entry_price = weighted_price
                position.current_price = price
                # Update SL/TP if provided
                if stop_loss is not None:
                    position.stop_loss = stop_loss
                if take_profit is not None:
                    position.take_profit = take_profit
        
        # Update last trade time
        self.last_trade_time[ticker] = datetime.utcnow()
        
        logger.info(f"Position updated for {ticker}: {self.positions.get(ticker)}")
    
    def _record_trade_pnl(self, pnl: float):
        """Record trade P&L and update performance metrics."""
        self.total_pnl += pnl
        self.daily_pnl += pnl
        self.total_trades += 1
        
        if pnl > 0:
            self.winning_trades += 1
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
        
        logger.info(f"Trade P&L recorded: ${pnl:.2f} (Total: ${self.total_pnl:.2f})")
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary."""
        
        # Calculate current portfolio value
        positions_value = sum(pos.market_value for pos in self.positions.values())
        cash = self.portfolio_value - positions_value
        total_value = cash + positions_value
        
        # Calculate unrealized P&L
        unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        
        # Calculate performance metrics
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        
        return {
            "portfolio_value": total_value,
            "cash": cash,
            "positions_value": positions_value,
            "unrealized_pnl": unrealized_pnl,
            "realized_pnl": self.total_pnl,
            "daily_pnl": self.daily_pnl,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "win_rate": win_rate,
            "consecutive_losses": self.consecutive_losses,
            "positions": {
                ticker: {
                    "quantity": pos.quantity,
                    "entry_price": pos.entry_price,
                    "current_price": pos.current_price,
                    "market_value": pos.market_value,
                    "unrealized_pnl": pos.unrealized_pnl,
                    "unrealized_pnl_percent": pos.unrealized_pnl_percent
                }
                for ticker, pos in self.positions.items()
            }
        }
    
    def get_risk_metrics(self) -> RiskMetrics:
        """Calculate comprehensive risk metrics."""
        
        positions_value = sum(abs(pos.market_value) for pos in self.positions.values())
        total_value = self.portfolio_value
        
        return RiskMetrics(
            portfolio_value=total_value,
            total_exposure=positions_value,
            max_position_risk=max(
                (abs(pos.market_value) / total_value for pos in self.positions.values()),
                default=0.0
            ),
            portfolio_beta=1.0,  # Simplified - would need market data
            var_95=total_value * 0.05,  # Simplified 5% VaR
            sharpe_ratio=0.0,  # Would need risk-free rate and volatility
            max_drawdown=abs(min(self.daily_pnl, 0)) / total_value,
            win_rate=self.winning_trades / self.total_trades if self.total_trades > 0 else 0,
            risk_adjusted_return=self.total_pnl / total_value if total_value > 0 else 0
        )
    
    def reset_daily_metrics(self):
        """Reset daily metrics (call at start of each trading day)."""
        self.daily_pnl = 0.0
        logger.info("Daily metrics reset")
    
    def should_halt_trading(self) -> bool:
        """Determine if trading should be halted due to risk conditions."""
        
        # Check daily loss limit
        if self.daily_pnl < -self.daily_loss_limit * self.portfolio_value:
            return True
        
        # Check consecutive losses
        if self.consecutive_losses >= 5:
            return True
        
        # Check portfolio drawdown
        if self.total_pnl < -0.2 * self.portfolio_value:  # 20% drawdown
            return True
        
        return False
