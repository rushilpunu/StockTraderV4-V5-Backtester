"""Alpaca paper trading API integration."""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import backoff
from loguru import logger

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import (
        MarketOrderRequest, LimitOrderRequest, StopOrderRequest,
        GetOrdersRequest, ClosePositionRequest
    )
    from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestQuoteRequest, StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
except ImportError:
    logger.error("Alpaca-py not installed. Install with: pip install alpaca-py")
    raise

from .config import config
from .trading_engine import TradingDecision, TradeAction, OrderType as EngineOrderType


@dataclass
class AlpacaOrder:
    """Container for Alpaca order information."""
    order_id: str
    ticker: str
    side: str
    quantity: int
    order_type: str
    status: str
    filled_price: Optional[float]
    filled_quantity: int
    submitted_at: datetime
    filled_at: Optional[datetime]
    
    @classmethod
    def from_alpaca_order(cls, order) -> 'AlpacaOrder':
        """Create AlpacaOrder from Alpaca API order object."""
        try:
            return cls(
                order_id=str(order.id),
                ticker=order.symbol,
                side=order.side.value,
                quantity=int(order.qty) if order.qty is not None else 0,
                order_type=order.order_type.value,
                status=order.status.value,
                filled_price=float(order.filled_avg_price) if order.filled_avg_price is not None else None,
                filled_quantity=int(order.filled_qty) if order.filled_qty is not None else 0,
                submitted_at=order.submitted_at,
                filled_at=order.filled_at
            )
        except Exception as e:
            logger.error(f"Error creating AlpacaOrder: {e}, order fields: qty={getattr(order, 'qty', 'MISSING')}, filled_qty={getattr(order, 'filled_qty', 'MISSING')}")
            raise


@dataclass
class AlpacaPosition:
    """Container for Alpaca position information."""
    ticker: str
    quantity: int
    market_value: float
    cost_basis: float
    unrealized_pnl: float
    unrealized_pnl_percent: float
    current_price: float
    
    @classmethod
    def from_alpaca_position(cls, position) -> 'AlpacaPosition':
        """Create AlpacaPosition from Alpaca API position object."""
        return cls(
            ticker=position.symbol,
            quantity=int(position.qty) if position.qty is not None else 0,
            market_value=float(position.market_value) if position.market_value is not None else 0.0,
            cost_basis=float(position.cost_basis) if position.cost_basis is not None else 0.0,
            unrealized_pnl=float(position.unrealized_pnl) if position.unrealized_pnl is not None else 0.0,
            unrealized_pnl_percent=float(position.unrealized_plpc) * 100 if position.unrealized_plpc is not None else 0.0,
            current_price=float(position.current_price) if position.current_price is not None else 0.0
        )


class AlpacaClient:
    """Alpaca paper trading API client with comprehensive order management."""
    
    def __init__(self):
        try:
            # Initialize trading client
            self.trading_client = TradingClient(
                api_key=config.alpaca_api_key,
                secret_key=config.alpaca_secret_key,
                paper=True  # Always use paper trading
            )
            
            # Initialize data client
            self.data_client = StockHistoricalDataClient(
                api_key=config.alpaca_api_key,
                secret_key=config.alpaca_secret_key
            )
            
            # Order tracking
            self.pending_orders: Dict[str, AlpacaOrder] = {}
            self.completed_orders: List[AlpacaOrder] = []
            
            logger.info("Alpaca client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Alpaca client: {e}")
            raise
    
    @backoff.on_exception(
        backoff.expo,
        Exception,
        max_tries=3,
        max_time=30
    )
    async def get_current_price(self, ticker: str) -> Optional[float]:
        """Get current price for a ticker."""
        try:
            request = StockLatestQuoteRequest(symbol_or_symbols=[ticker])
            quotes = self.data_client.get_stock_latest_quote(request)
            
            if ticker in quotes:
                quote = quotes[ticker]
                # Use mid price between bid and ask
                return (float(quote.bid_price) + float(quote.ask_price)) / 2
            
            logger.warning(f"No quote data available for {ticker}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting current price for {ticker}: {e}")
            return None
    
    @backoff.on_exception(
        backoff.expo,
        Exception,
        max_tries=3,
        max_time=30
    )
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information."""
        try:
            account = self.trading_client.get_account()
            
            return {
                "account_id": account.id,
                "status": account.status.value,
                "currency": account.currency,
                "buying_power": float(account.buying_power),
                "cash": float(account.cash),
                "portfolio_value": float(account.portfolio_value),
                "equity": float(account.equity),
                "initial_margin": float(account.initial_margin),
                "maintenance_margin": float(account.maintenance_margin),
                "daytrade_count": account.daytrade_count,
                "sma": float(account.sma) if account.sma else 0.0,
                "created_at": account.created_at
            }
            
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return {}
    
    @backoff.on_exception(
        backoff.expo,
        Exception,
        max_tries=3,
        max_time=30
    )
    async def execute_trading_decision(self, decision: TradingDecision) -> Optional[str]:
        """
        Execute a trading decision through Alpaca API.
        
        Args:
            decision: TradingDecision object
            
        Returns:
            Order ID if successful, None otherwise
        """
        if not config.trading_enabled:
            logger.info(f"Trading disabled - would execute: {decision.to_dict()}")
            return None
        
        try:
            # Get current price if not provided
            if not decision.price:
                current_price = await self.get_current_price(decision.ticker)
                if not current_price:
                    logger.error(f"Cannot get current price for {decision.ticker}")
                    return None
            else:
                current_price = decision.price
            
            # Convert action to Alpaca side
            if decision.action in [TradeAction.BUY, TradeAction.CLOSE]:
                side = OrderSide.BUY
            else:
                side = OrderSide.SELL
            
            # Create order request based on type
            if decision.order_type == EngineOrderType.MARKET:
                order_request = MarketOrderRequest(
                    symbol=decision.ticker,
                    qty=decision.quantity,
                    side=side,
                    time_in_force=TimeInForce.DAY
                )
                
            elif decision.order_type == EngineOrderType.LIMIT:
                # For limit orders, use current price as limit price
                limit_price = current_price
                order_request = LimitOrderRequest(
                    symbol=decision.ticker,
                    qty=decision.quantity,
                    side=side,
                    time_in_force=TimeInForce.DAY,
                    limit_price=limit_price
                )
                
            else:
                logger.error(f"Unsupported order type: {decision.order_type}")
                return None
            
            # Submit order
            order = self.trading_client.submit_order(order_request)
            order_id = str(order.id)
            
            # Track the order
            alpaca_order = AlpacaOrder.from_alpaca_order(order)
            self.pending_orders[order_id] = alpaca_order
            
            logger.info(
                f"Order submitted: {side.value} {decision.quantity} {decision.ticker} "
                f"(Order ID: {order_id})"
            )
            
            # Submit stop loss and take profit orders if specified
            await self._submit_bracket_orders(decision, order_id, current_price)
            
            return order_id
            
        except Exception as e:
            logger.error(f"Error executing trading decision: {e}")
            return None
    
    async def _submit_bracket_orders(
        self, 
        decision: TradingDecision, 
        parent_order_id: str,
        current_price: float
    ):
        """Submit stop loss and take profit orders."""
        try:
            # Wait a bit for the parent order to potentially fill
            await asyncio.sleep(2)
            
            # Check if parent order was filled
            parent_order = await self.get_order_status(parent_order_id)
            if not parent_order or parent_order.status != "filled":
                logger.info(f"Parent order {parent_order_id} not filled yet, skipping bracket orders")
                return
            
            # Determine quantity and side for bracket orders
            if decision.action in [TradeAction.BUY, TradeAction.CLOSE]:
                bracket_side = OrderSide.SELL  # Close long position
            else:
                bracket_side = OrderSide.BUY   # Close short position
            
            # Submit stop loss order
            if decision.stop_loss:
                try:
                    stop_order_request = StopOrderRequest(
                        symbol=decision.ticker,
                        qty=decision.quantity,
                        side=bracket_side,
                        time_in_force=TimeInForce.GTC,
                        stop_price=decision.stop_loss
                    )
                    
                    stop_order = self.trading_client.submit_order(stop_order_request)
                    logger.info(f"Stop loss order submitted: {stop_order.id} at ${decision.stop_loss}")
                    
                except Exception as e:
                    logger.error(f"Error submitting stop loss order: {e}")
            
            # Submit take profit order
            if decision.take_profit:
                try:
                    profit_order_request = LimitOrderRequest(
                        symbol=decision.ticker,
                        qty=decision.quantity,
                        side=bracket_side,
                        time_in_force=TimeInForce.GTC,
                        limit_price=decision.take_profit
                    )
                    
                    profit_order = self.trading_client.submit_order(profit_order_request)
                    logger.info(f"Take profit order submitted: {profit_order.id} at ${decision.take_profit}")
                    
                except Exception as e:
                    logger.error(f"Error submitting take profit order: {e}")
                    
        except Exception as e:
            logger.error(f"Error submitting bracket orders: {e}")
    
    @backoff.on_exception(
        backoff.expo,
        Exception,
        max_tries=3,
        max_time=30
    )
    async def get_order_status(self, order_id: str) -> Optional[AlpacaOrder]:
        """Get status of a specific order."""
        try:
            order = self.trading_client.get_order_by_id(order_id)
            alpaca_order = AlpacaOrder.from_alpaca_order(order)
            
            # Update our tracking
            if order_id in self.pending_orders:
                self.pending_orders[order_id] = alpaca_order
                
                # Move to completed if filled or cancelled
                if alpaca_order.status in ["filled", "cancelled", "rejected"]:
                    self.completed_orders.append(alpaca_order)
                    del self.pending_orders[order_id]
            
            return alpaca_order
            
        except Exception as e:
            logger.error(f"Error getting order status for {order_id}: {e}")
            return None
    
    @backoff.on_exception(
        backoff.expo,
        Exception,
        max_tries=3,
        max_time=30
    )
    async def get_positions(self) -> List[AlpacaPosition]:
        """Get all current positions."""
        try:
            positions = self.trading_client.get_all_positions()
            return [AlpacaPosition.from_alpaca_position(pos) for pos in positions]
            
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []
    
    @backoff.on_exception(
        backoff.expo,
        Exception,
        max_tries=3,
        max_time=30
    )
    async def close_position(self, ticker: str, percentage: float = 1.0) -> Optional[str]:
        """
        Close a position (partially or completely).
        
        Args:
            ticker: Stock ticker
            percentage: Percentage of position to close (0.0 to 1.0)
            
        Returns:
            Order ID if successful, None otherwise
        """
        try:
            if percentage == 1.0:
                # Close entire position
                close_request = ClosePositionRequest(
                    percentage=percentage
                )
                order = self.trading_client.close_position(ticker, close_request)
            else:
                # Close partial position - need to calculate quantity
                positions = await self.get_positions()
                position = next((p for p in positions if p.ticker == ticker), None)
                
                if not position:
                    logger.warning(f"No position found for {ticker}")
                    return None
                
                quantity_to_close = int(abs(position.quantity) * percentage)
                side = OrderSide.SELL if position.quantity > 0 else OrderSide.BUY
                
                order_request = MarketOrderRequest(
                    symbol=ticker,
                    qty=quantity_to_close,
                    side=side,
                    time_in_force=TimeInForce.DAY
                )
                
                order = self.trading_client.submit_order(order_request)
            
            order_id = str(order.id)
            logger.info(f"Position close order submitted for {ticker}: {order_id}")
            
            return order_id
            
        except Exception as e:
            logger.error(f"Error closing position for {ticker}: {e}")
            return None
    
    @backoff.on_exception(
        backoff.expo,
        Exception,
        max_tries=3,
        max_time=30
    )
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        try:
            self.trading_client.cancel_order_by_id(order_id)
            logger.info(f"Order cancelled: {order_id}")
            
            # Update our tracking
            if order_id in self.pending_orders:
                self.pending_orders[order_id].status = "cancelled"
                self.completed_orders.append(self.pending_orders[order_id])
                del self.pending_orders[order_id]
            
            return True
            
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    @backoff.on_exception(
        backoff.expo,
        Exception,
        max_tries=3,
        max_time=30
    )
    async def get_recent_orders(self, limit: int = 50) -> List[AlpacaOrder]:
        """Get recent orders."""
        try:
            request = GetOrdersRequest(
                status="all",
                limit=limit,
                nested=True
            )
            
            orders = self.trading_client.get_orders(request)
            return [AlpacaOrder.from_alpaca_order(order) for order in orders]
            
        except Exception as e:
            logger.error(f"Error getting recent orders: {e}")
            return []
    
    async def get_portfolio_performance(self, days_back: int = 30) -> Dict[str, Any]:
        """Get portfolio performance metrics."""
        try:
            account = await self.get_account_info()
            positions = await self.get_positions()
            recent_orders = await self.get_recent_orders(100)
            
            # Calculate basic metrics
            total_value = account.get("portfolio_value", 0)
            cash = account.get("cash", 0)
            equity = account.get("equity", 0)
            
            # Calculate position metrics
            total_unrealized_pnl = sum(pos.unrealized_pnl for pos in positions)
            
            # Calculate trade metrics from recent orders
            filled_orders = [order for order in recent_orders if order.status == "filled"]
            total_trades = len(filled_orders)
            
            # Calculate win rate (simplified)
            winning_positions = len([pos for pos in positions if pos.unrealized_pnl > 0])
            total_positions = len(positions)
            win_rate = (winning_positions / total_positions * 100) if total_positions > 0 else 0
            
            return {
                "portfolio_value": total_value,
                "cash": cash,
                "equity": equity,
                "positions_count": len(positions),
                "total_unrealized_pnl": total_unrealized_pnl,
                "total_trades": total_trades,
                "win_rate": win_rate,
                "buying_power": account.get("buying_power", 0),
                "day_trade_count": account.get("daytrade_count", 0),
                "positions": [
                    {
                        "ticker": pos.ticker,
                        "quantity": pos.quantity,
                        "market_value": pos.market_value,
                        "unrealized_pnl": pos.unrealized_pnl,
                        "unrealized_pnl_percent": pos.unrealized_pnl_percent
                    }
                    for pos in positions
                ]
            }
            
        except Exception as e:
            logger.error(f"Error getting portfolio performance: {e}")
            return {}
    
    async def update_pending_orders(self):
        """Update status of all pending orders."""
        for order_id in list(self.pending_orders.keys()):
            await self.get_order_status(order_id)
    
    async def get_market_hours(self) -> Dict[str, Any]:
        """Get market hours information."""
        try:
            clock = self.trading_client.get_clock()
            
            return {
                "timestamp": clock.timestamp,
                "is_open": clock.is_open,
                "next_open": clock.next_open,
                "next_close": clock.next_close
            }
            
        except Exception as e:
            logger.error(f"Error getting market hours: {e}")
            return {}
    
    def get_order_summary(self) -> Dict[str, Any]:
        """Get summary of order activity."""
        return {
            "pending_orders": len(self.pending_orders),
            "completed_orders": len(self.completed_orders),
            "pending_order_details": [
                {
                    "order_id": order.order_id,
                    "ticker": order.ticker,
                    "side": order.side,
                    "quantity": order.quantity,
                    "status": order.status,
                    "submitted_at": order.submitted_at.isoformat()
                }
                for order in self.pending_orders.values()
            ],
            "recent_completed_orders": [
                {
                    "order_id": order.order_id,
                    "ticker": order.ticker,
                    "side": order.side,
                    "quantity": order.quantity,
                    "status": order.status,
                    "filled_price": order.filled_price,
                    "filled_at": order.filled_at.isoformat() if order.filled_at else None
                }
                for order in self.completed_orders[-10:]  # Last 10 completed orders
            ]
        }
