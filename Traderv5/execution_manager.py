"""Execution manager for converting trade decisions into broker orders."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

from Traderv5.cash_ledger import CashLedger, CashTransaction
from Traderv5.decision_router import TradeDecision, TradeAction, TradeHorizon
from Traderv5.risk_manager import RiskManager, RiskMetrics

_LOG = logging.getLogger("traderv5.execution_manager")


class OrderType(Enum):
    """Order types."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    BRACKET = "BRACKET"


class OrderStatus(Enum):
    """Order status."""
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class OrderSide(Enum):
    """Order side."""
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class Order:
    """Represents a trading order."""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    price: Optional[Decimal]
    stop_price: Optional[Decimal]
    limit_price: Optional[Decimal]
    status: OrderStatus
    filled_quantity: int = 0
    filled_price: Optional[Decimal] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    parent_order_id: Optional[str] = None
    child_order_ids: List[str] = field(default_factory=list)
    
    @property
    def is_filled(self) -> bool:
        return self.status == OrderStatus.FILLED
    
    @property
    def is_pending(self) -> bool:
        return self.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED]
    
    @property
    def remaining_quantity(self) -> int:
        return self.quantity - self.filled_quantity
    
    @property
    def is_bracket_order(self) -> bool:
        return self.order_type == OrderType.BRACKET or self.child_order_ids


@dataclass
class BracketOrder:
    """Represents a bracket order with entry, stop, and take profit."""
    bracket_id: str
    symbol: str
    entry_order: Order
    stop_order: Optional[Order] = None
    take_profit_order: Optional[Order] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def is_complete(self) -> bool:
        """Check if bracket order is complete (entry filled and exits placed)."""
        return (
            self.entry_order.is_filled and
            self.stop_order is not None and
            self.take_profit_order is not None
        )


@dataclass
class ExecutionResult:
    """Result of order execution."""
    success: bool
    order_id: str
    symbol: str
    side: OrderSide
    quantity: int
    price: Decimal
    timestamp: datetime
    message: str
    cash_transaction: Optional[CashTransaction] = None
    risk_metrics: Optional[RiskMetrics] = None


class ExecutionManager:
    """
    Manages order execution and ensures compliance with cash account rules.
    
    Features:
    - Converts trade decisions to orders
    - Creates bracket orders (entry + stop + take profit)
    - Manages cash ledger updates
    - Prevents good faith violations
    - Tracks order status
    """
    
    def __init__(
        self,
        cash_ledger: CashLedger,
        risk_manager: RiskManager,
    ):
        self.cash_ledger = cash_ledger
        self.risk_manager = risk_manager
        self.orders: Dict[str, Order] = {}
        self.bracket_orders: Dict[str, BracketOrder] = {}
        self.execution_history: List[ExecutionResult] = []
        
        _LOG.info("Initialized execution manager")
    
    def execute_trade_decision(
        self,
        decision: TradeDecision,
        atr: Decimal,
        current_price: Decimal,
    ) -> ExecutionResult:
        """
        Execute a trade decision by creating appropriate orders.
        
        Args:
            decision: Trade decision from router
            atr: Average True Range for risk calculations
            current_price: Current market price
            
        Returns:
            ExecutionResult with execution details
        """
        if decision.action == TradeAction.BUY:
            return self._execute_buy_decision(decision, atr, current_price)
        elif decision.action == TradeAction.SELL:
            return self._execute_sell_decision(decision, current_price)
        else:
            return ExecutionResult(
                success=False,
                order_id="",
                symbol=decision.symbol,
                side=OrderSide.BUY if decision.action == TradeAction.BUY else OrderSide.SELL,
                quantity=0,
                price=current_price,
                timestamp=datetime.utcnow(),
                message=f"No execution needed for action: {decision.action.value}",
            )
    
    def _execute_buy_decision(
        self,
        decision: TradeDecision,
        atr: Decimal,
        current_price: Decimal,
    ) -> ExecutionResult:
        """Execute a buy decision with bracket order."""
        try:
            # Create entry order
            entry_order = self._create_entry_order(decision, current_price)
            
            # Create bracket order if we have stop/take profit levels
            if decision.stop_loss and decision.take_profit:
                bracket_order = self._create_bracket_order(entry_order, decision, atr)
                self.bracket_orders[bracket_order.bracket_id] = bracket_order
                
                # Simulate order execution (in real implementation, submit to broker)
                execution_result = self._simulate_order_execution(entry_order, current_price)
                
                if execution_result.success:
                    # Update cash ledger
                    cash_transaction = self.cash_ledger.execute_buy(
                        symbol=decision.symbol,
                        quantity=decision.quantity,
                        price=current_price,
                        transaction_id=entry_order.order_id,
                    )
                    
                    # Create risk metrics
                    risk_metrics = self.risk_manager.create_risk_metrics(
                        symbol=decision.symbol,
                        entry_price=current_price,
                        quantity=decision.quantity,
                        atr=atr,
                    )
                    
                    execution_result.cash_transaction = cash_transaction
                    execution_result.risk_metrics = risk_metrics
                
                return execution_result
            else:
                # Simple market order without bracket
                execution_result = self._simulate_order_execution(entry_order, current_price)
                
                if execution_result.success:
                    cash_transaction = self.cash_ledger.execute_buy(
                        symbol=decision.symbol,
                        quantity=decision.quantity,
                        price=current_price,
                        transaction_id=entry_order.order_id,
                    )
                    execution_result.cash_transaction = cash_transaction
                
                return execution_result
                
        except Exception as e:
            _LOG.error("Failed to execute buy decision for %s: %s", decision.symbol, e)
            return ExecutionResult(
                success=False,
                order_id="",
                symbol=decision.symbol,
                side=OrderSide.BUY,
                quantity=decision.quantity,
                price=current_price,
                timestamp=datetime.utcnow(),
                message=f"Execution failed: {str(e)}",
            )
    
    def _execute_sell_decision(
        self,
        decision: TradeDecision,
        current_price: Decimal,
    ) -> ExecutionResult:
        """Execute a sell decision."""
        try:
            # Create sell order
            sell_order = self._create_sell_order(decision, current_price)
            
            # Simulate order execution
            execution_result = self._simulate_order_execution(sell_order, current_price)
            
            if execution_result.success:
                # Update cash ledger
                cash_transaction = self.cash_ledger.execute_sell(
                    symbol=decision.symbol,
                    quantity=decision.quantity,
                    price=current_price,
                    transaction_id=sell_order.order_id,
                )
                execution_result.cash_transaction = cash_transaction
            
            return execution_result
            
        except Exception as e:
            _LOG.error("Failed to execute sell decision for %s: %s", decision.symbol, e)
            return ExecutionResult(
                success=False,
                order_id="",
                symbol=decision.symbol,
                side=OrderSide.SELL,
                quantity=decision.quantity,
                price=current_price,
                timestamp=datetime.utcnow(),
                message=f"Execution failed: {str(e)}",
            )
    
    def _create_entry_order(self, decision: TradeDecision, current_price: Decimal) -> Order:
        """Create an entry order from trade decision."""
        order_id = str(uuid4())
        
        order = Order(
            order_id=order_id,
            symbol=decision.symbol,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,  # Use market orders for simplicity
            quantity=decision.quantity,
            price=current_price,
            stop_price=None,
            limit_price=None,
            status=OrderStatus.PENDING,
        )
        
        self.orders[order_id] = order
        return order
    
    def _create_sell_order(self, decision: TradeDecision, current_price: Decimal) -> Order:
        """Create a sell order from trade decision."""
        order_id = str(uuid4())
        
        order = Order(
            order_id=order_id,
            symbol=decision.symbol,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=decision.quantity,
            price=current_price,
            stop_price=None,
            limit_price=None,
            status=OrderStatus.PENDING,
        )
        
        self.orders[order_id] = order
        return order
    
    def _create_bracket_order(
        self,
        entry_order: Order,
        decision: TradeDecision,
        atr: Decimal,
    ) -> BracketOrder:
        """Create a bracket order with entry, stop, and take profit."""
        bracket_id = str(uuid4())
        
        # Create stop loss order
        stop_order = Order(
            order_id=str(uuid4()),
            symbol=decision.symbol,
            side=OrderSide.SELL,
            order_type=OrderType.STOP,
            quantity=decision.quantity,
            price=None,
            stop_price=decision.stop_loss,
            limit_price=None,
            status=OrderStatus.PENDING,
            parent_order_id=entry_order.order_id,
        )
        
        # Create take profit order
        take_profit_order = Order(
            order_id=str(uuid4()),
            symbol=decision.symbol,
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=decision.quantity,
            price=decision.take_profit,
            stop_price=None,
            limit_price=decision.take_profit,
            status=OrderStatus.PENDING,
            parent_order_id=entry_order.order_id,
        )
        
        # Store child orders
        entry_order.child_order_ids = [stop_order.order_id, take_profit_order.order_id]
        self.orders[stop_order.order_id] = stop_order
        self.orders[take_profit_order.order_id] = take_profit_order
        
        bracket_order = BracketOrder(
            bracket_id=bracket_id,
            symbol=decision.symbol,
            entry_order=entry_order,
            stop_order=stop_order,
            take_profit_order=take_profit_order,
        )
        
        return bracket_order
    
    def _simulate_order_execution(self, order: Order, current_price: Decimal) -> ExecutionResult:
        """Simulate order execution (replace with real broker API in production)."""
        # Simulate immediate fill for market orders
        if order.order_type == OrderType.MARKET:
            order.status = OrderStatus.FILLED
            order.filled_quantity = order.quantity
            order.filled_price = current_price
            order.updated_at = datetime.utcnow()
            
            return ExecutionResult(
                success=True,
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                price=current_price,
                timestamp=datetime.utcnow(),
                message="Order filled successfully",
            )
        else:
            return ExecutionResult(
                success=False,
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                price=current_price,
                timestamp=datetime.utcnow(),
                message="Order type not supported in simulation",
            )
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        if order.is_pending:
            order.status = OrderStatus.CANCELLED
            order.updated_at = datetime.utcnow()
            
            # Release reserved cash if it was a buy order
            if order.side == OrderSide.BUY and order.is_filled:
                # In a real implementation, you'd need to track reserved cash per order
                pass
            
            _LOG.info("Cancelled order %s", order_id)
            return True
        
        return False
    
    def get_order_status(self, order_id: str) -> Optional[Order]:
        """Get order status by ID."""
        return self.orders.get(order_id)
    
    def get_active_orders(self) -> List[Order]:
        """Get all active (pending/submitted) orders."""
        return [order for order in self.orders.values() if order.is_pending]
    
    def get_bracket_order(self, bracket_id: str) -> Optional[BracketOrder]:
        """Get bracket order by ID."""
        return self.bracket_orders.get(bracket_id)
    
    def get_execution_history(self, symbol: Optional[str] = None) -> List[ExecutionResult]:
        """Get execution history, optionally filtered by symbol."""
        if symbol is None:
            return self.execution_history
        
        return [result for result in self.execution_history if result.symbol == symbol]
    
    def check_good_faith_violation(
        self,
        symbol: str,
        quantity: int,
        price: Decimal,
    ) -> Tuple[bool, str]:
        """
        Check if a trade would cause a good faith violation.
        
        In a cash account, a good faith violation occurs when you:
        1. Buy a security and sell it before the purchase settles (T+2)
        2. Use the proceeds from the sale to buy another security
        
        Args:
            symbol: Stock symbol
            quantity: Quantity to trade
            price: Price per share
            
        Returns:
            Tuple of (would_violate, reason)
        """
        # This is a simplified check - in practice, you'd need to track
        # all recent transactions and their settlement dates
        
        total_cost = Decimal(str(quantity)) * price
        
        # Check if we have enough settled cash
        can_buy, reason = self.cash_ledger.can_buy(total_cost)
        if not can_buy:
            return True, f"Good faith violation risk: {reason}"
        
        return False, "No good faith violation risk"
    
    def get_execution_summary(self) -> Dict[str, any]:
        """Get summary of execution manager state."""
        active_orders = self.get_active_orders()
        filled_orders = [order for order in self.orders.values() if order.is_filled]
        
        return {
            "total_orders": len(self.orders),
            "active_orders": len(active_orders),
            "filled_orders": len(filled_orders),
            "bracket_orders": len(self.bracket_orders),
            "execution_history_count": len(self.execution_history),
            "cash_summary": self.cash_ledger.get_cash_summary(),
        }


__all__ = [
    "ExecutionManager",
    "Order",
    "BracketOrder",
    "ExecutionResult",
    "OrderType",
    "OrderStatus",
    "OrderSide",
]
