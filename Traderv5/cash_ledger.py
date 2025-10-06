"""Cash ledger system for tracking settled vs unsettled funds in cash accounts."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Tuple

from Traderv5.calendars import TradingCalendar, calculate_settlement_date
from Traderv5.configuration import CalendarSettings

_LOG = logging.getLogger("traderv5.cash_ledger")


@dataclass
class CashTransaction:
    """Represents a cash transaction with settlement tracking."""
    transaction_id: str
    symbol: str
    transaction_type: str  # "BUY" or "SELL"
    amount: Decimal
    trade_date: date
    settlement_date: date
    quantity: int
    price: Decimal
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def is_buy(self) -> bool:
        return self.transaction_type.upper() == "BUY"
    
    @property
    def is_sell(self) -> bool:
        return self.transaction_type.upper() == "SELL"


@dataclass
class CashLedgerState:
    """Current state of the cash ledger."""
    settled_cash: Decimal
    reserved_cash: Decimal
    unsettled_cash_by_date: Dict[date, Decimal]
    transactions: List[CashTransaction]
    equity: Decimal
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def available_cash(self) -> Decimal:
        """Cash available for new purchases (settled - reserved)."""
        return self.settled_cash - self.reserved_cash
    
    @property
    def total_unsettled_cash(self) -> Decimal:
        """Total cash that will become available on future settlement dates."""
        return sum(self.unsettled_cash_by_date.values())
    
    @property
    def total_cash(self) -> Decimal:
        """Total cash including settled, reserved, and unsettled."""
        return self.settled_cash + self.total_unsettled_cash


class CashLedger:
    """Manages cash balances for cash accounts with T+1 settlement rules."""
    
    def __init__(self, starting_equity: float, calendar_settings: CalendarSettings):
        self.calendar = TradingCalendar(calendar_settings)
        self.starting_equity = Decimal(str(starting_equity))
        
        # Initialize ledger state
        self.state = CashLedgerState(
            settled_cash=self.starting_equity,
            reserved_cash=Decimal('0'),
            unsettled_cash_by_date={},
            transactions=[],
            equity=self.starting_equity,
        )
        
        _LOG.info("Initialized cash ledger with starting equity: $%.2f", starting_equity)
    
    def can_buy(self, amount: Decimal, trade_date: Optional[date] = None) -> Tuple[bool, str]:
        """
        Check if a buy order can be executed with available settled cash.
        
        Returns:
            Tuple of (can_buy, reason)
        """
        if amount <= 0:
            return False, "Buy amount must be positive"
        
        available = self.state.available_cash
        if amount > available:
            return False, f"Insufficient settled cash. Need ${amount}, have ${available}"
        
        return True, "Sufficient settled cash available"
    
    def can_sell(self, symbol: str, quantity: int, price: Decimal) -> Tuple[bool, str]:
        """
        Check if a sell order can be executed.
        
        For cash accounts, we can always sell if we have the position.
        This method is mainly for validation and logging.
        """
        if quantity <= 0:
            return False, "Sell quantity must be positive"
        
        if price <= 0:
            return False, "Sell price must be positive"
        
        # In a real implementation, you'd check position here
        # For now, we assume we can sell if we have the position
        return True, "Sell order can be executed"
    
    def execute_buy(
        self,
        symbol: str,
        quantity: int,
        price: Decimal,
        trade_date: Optional[date] = None,
        transaction_id: Optional[str] = None,
    ) -> CashTransaction:
        """
        Execute a buy order and update cash balances.
        
        Args:
            symbol: Stock symbol
            quantity: Number of shares
            price: Price per share
            trade_date: Trade date (defaults to today)
            transaction_id: Optional transaction ID
            
        Returns:
            CashTransaction object
            
        Raises:
            ValueError: If insufficient settled cash
        """
        if trade_date is None:
            trade_date = date.today()
        
        if transaction_id is None:
            transaction_id = f"BUY_{symbol}_{trade_date}_{len(self.state.transactions)}"
        
        total_cost = Decimal(str(quantity)) * price
        total_cost = total_cost.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
        
        # Check if we can afford this purchase
        can_buy, reason = self.can_buy(total_cost, trade_date)
        if not can_buy:
            raise ValueError(f"Cannot execute buy order: {reason}")
        
        # Calculate settlement date
        settlement_date = calculate_settlement_date(
            trade_date,
            settings=self.calendar.settings,
            settlement_lag=1,  # T+1 settlement
        )
        
        # Create transaction
        transaction = CashTransaction(
            transaction_id=transaction_id,
            symbol=symbol,
            transaction_type="BUY",
            amount=total_cost,
            trade_date=trade_date,
            settlement_date=settlement_date,
            quantity=quantity,
            price=price,
        )
        
        # Update cash balances
        # For filled buys, reduce settled cash immediately. Do not increase
        # reserved cash here, as reservation should only be used for pending
        # orders. This prevents available cash from being permanently reduced.
        self.state.settled_cash -= total_cost
        self.state.transactions.append(transaction)
        self.state.last_updated = datetime.utcnow()
        
        _LOG.info(
            "Executed BUY: %s %d shares @ $%.2f = $%.2f (settles %s)",
            symbol, quantity, price, total_cost, settlement_date
        )
        
        return transaction
    
    def execute_sell(
        self,
        symbol: str,
        quantity: int,
        price: Decimal,
        trade_date: Optional[date] = None,
        transaction_id: Optional[str] = None,
    ) -> CashTransaction:
        """
        Execute a sell order and update cash balances.
        
        Args:
            symbol: Stock symbol
            quantity: Number of shares
            price: Price per share
            trade_date: Trade date (defaults to today)
            transaction_id: Optional transaction ID
            
        Returns:
            CashTransaction object
        """
        if trade_date is None:
            trade_date = date.today()
        
        if transaction_id is None:
            transaction_id = f"SELL_{symbol}_{trade_date}_{len(self.state.transactions)}"
        
        total_proceeds = Decimal(str(quantity)) * price
        total_proceeds = total_proceeds.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
        
        # Calculate settlement date
        settlement_date = calculate_settlement_date(
            trade_date,
            settings=self.calendar.settings,
            settlement_lag=1,  # T+1 settlement
        )
        
        # Create transaction
        transaction = CashTransaction(
            transaction_id=transaction_id,
            symbol=symbol,
            transaction_type="SELL",
            amount=total_proceeds,
            trade_date=trade_date,
            settlement_date=settlement_date,
            quantity=quantity,
            price=price,
        )
        
        # Update cash balances - proceeds go to unsettled
        if settlement_date not in self.state.unsettled_cash_by_date:
            self.state.unsettled_cash_by_date[settlement_date] = Decimal('0')
        self.state.unsettled_cash_by_date[settlement_date] += total_proceeds
        self.state.transactions.append(transaction)
        self.state.last_updated = datetime.utcnow()
        
        _LOG.info(
            "Executed SELL: %s %d shares @ $%.2f = $%.2f (settles %s)",
            symbol, quantity, price, total_proceeds, settlement_date
        )
        
        return transaction
    
    def settle_cash(self, settlement_date: date) -> Decimal:
        """
        Move unsettled cash to settled cash for a specific settlement date.
        
        Args:
            settlement_date: Date to settle
            
        Returns:
            Amount settled
        """
        if settlement_date not in self.state.unsettled_cash_by_date:
            return Decimal('0')
        
        amount = self.state.unsettled_cash_by_date[settlement_date]
        self.state.settled_cash += amount
        del self.state.unsettled_cash_by_date[settlement_date]
        self.state.last_updated = datetime.utcnow()
        
        _LOG.info("Settled $%.2f for date %s", amount, settlement_date)
        return amount
    
    def end_of_day_settlement(self, current_date: date) -> Decimal:
        """
        Process end-of-day settlement for all cash due on the current date.
        
        Args:
            current_date: Current trading date
            
        Returns:
            Total amount settled
        """
        total_settled = Decimal('0')
        
        # Find all settlement dates that are today or in the past
        dates_to_settle = [
            settlement_date for settlement_date in self.state.unsettled_cash_by_date.keys()
            if settlement_date <= current_date
        ]
        
        for settlement_date in dates_to_settle:
            amount = self.settle_cash(settlement_date)
            total_settled += amount
        
        if total_settled > 0:
            _LOG.info("End-of-day settlement: $%.2f total settled", total_settled)
        
        return total_settled
    
    def release_reserved_cash(self, amount: Decimal, reason: str = "Order cancelled") -> None:
        """
        Release reserved cash back to settled cash.
        
        Args:
            amount: Amount to release
            reason: Reason for release
        """
        if amount > self.state.reserved_cash:
            raise ValueError(f"Cannot release ${amount}, only ${self.state.reserved_cash} reserved")
        
        self.state.reserved_cash -= amount
        self.state.settled_cash += amount
        self.state.last_updated = datetime.utcnow()
        
        _LOG.info("Released $%.2f reserved cash: %s", amount, reason)
    
    def get_cash_summary(self) -> Dict[str, float]:
        """Get a summary of current cash positions."""
        return {
            "settled_cash": float(self.state.settled_cash),
            "reserved_cash": float(self.state.reserved_cash),
            "available_cash": float(self.state.available_cash),
            "unsettled_cash": float(self.state.total_unsettled_cash),
            "total_cash": float(self.state.total_cash),
            "equity": float(self.state.equity),
        }
    
    def get_unsettled_schedule(self) -> Dict[str, float]:
        """Get schedule of when unsettled cash will become available."""
        return {
            settlement_date.isoformat(): float(amount)
            for settlement_date, amount in sorted(self.state.unsettled_cash_by_date.items())
        }
    
    def update_equity(self, new_equity: Decimal) -> None:
        """Update the equity value (e.g., after position revaluation)."""
        self.state.equity = new_equity
        self.state.last_updated = datetime.utcnow()


__all__ = [
    "CashLedger",
    "CashLedgerState", 
    "CashTransaction",
]
