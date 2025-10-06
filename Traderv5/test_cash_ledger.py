"""Unit tests for cash ledger system."""

import pytest
from datetime import date, datetime
from decimal import Decimal

from Traderv5.cash_ledger import CashLedger, CashTransaction
from Traderv5.configuration import CalendarSettings


class TestCashLedger:
    """Test cases for CashLedger."""
    
    @pytest.fixture
    def calendar_settings(self):
        """Create test calendar settings."""
        return CalendarSettings(
            exchange="XNYS",
            holiday_calendar="XNYS",
            custom_holidays=[],
            settlement_lag_days=1,
        )
    
    @pytest.fixture
    def cash_ledger(self, calendar_settings):
        """Create test cash ledger."""
        return CashLedger(25000.0, calendar_settings)
    
    def test_initialization(self, cash_ledger):
        """Test cash ledger initialization."""
        assert cash_ledger.state.settled_cash == Decimal('25000.00')
        assert cash_ledger.state.reserved_cash == Decimal('0')
        assert cash_ledger.state.total_unsettled_cash == Decimal('0')
        assert cash_ledger.state.equity == Decimal('25000.00')
    
    def test_can_buy_sufficient_cash(self, cash_ledger):
        """Test can_buy with sufficient cash."""
        can_buy, reason = cash_ledger.can_buy(Decimal('1000.00'))
        assert can_buy is True
        assert "Sufficient settled cash available" in reason
    
    def test_can_buy_insufficient_cash(self, cash_ledger):
        """Test can_buy with insufficient cash."""
        can_buy, reason = cash_ledger.can_buy(Decimal('30000.00'))
        assert can_buy is False
        assert "Insufficient settled cash" in reason
    
    def test_can_buy_negative_amount(self, cash_ledger):
        """Test can_buy with negative amount."""
        can_buy, reason = cash_ledger.can_buy(Decimal('-100.00'))
        assert can_buy is False
        assert "Buy amount must be positive" in reason
    
    def test_execute_buy_success(self, cash_ledger):
        """Test successful buy execution."""
        transaction = cash_ledger.execute_buy(
            symbol="AAPL",
            quantity=10,
            price=Decimal('150.00'),
            trade_date=date(2024, 1, 15),
        )
        
        assert transaction.symbol == "AAPL"
        assert transaction.quantity == 10
        assert transaction.price == Decimal('150.00')
        assert transaction.amount == Decimal('1500.00')
        assert transaction.is_buy is True
        
        # Check cash ledger state
        assert cash_ledger.state.settled_cash == Decimal('23500.00')  # 25000 - 1500
        assert cash_ledger.state.reserved_cash == Decimal('1500.00')
        assert cash_ledger.state.available_cash == Decimal('22000.00')  # 23500 - 1500
    
    def test_execute_buy_insufficient_cash(self, cash_ledger):
        """Test buy execution with insufficient cash."""
        with pytest.raises(ValueError, match="Cannot execute buy order"):
            cash_ledger.execute_buy(
                symbol="AAPL",
                quantity=200,
                price=Decimal('150.00'),
            )
    
    def test_execute_sell_success(self, cash_ledger):
        """Test successful sell execution."""
        # First buy some shares
        cash_ledger.execute_buy("AAPL", 10, Decimal('150.00'), date(2024, 1, 15))
        
        # Then sell them
        transaction = cash_ledger.execute_sell(
            symbol="AAPL",
            quantity=10,
            price=Decimal('160.00'),
            trade_date=date(2024, 1, 16),
        )
        
        assert transaction.symbol == "AAPL"
        assert transaction.quantity == 10
        assert transaction.price == Decimal('160.00')
        assert transaction.amount == Decimal('1600.00')
        assert transaction.is_sell is True
        
        # Check that proceeds go to unsettled cash
        settlement_date = transaction.settlement_date
        assert cash_ledger.state.unsettled_cash_by_date[settlement_date] == Decimal('1600.00')
    
    def test_settlement_date_calculation(self, cash_ledger):
        """Test settlement date calculation."""
        # Friday trade should settle on Monday (T+1)
        friday = date(2024, 1, 12)  # Friday
        settlement_date = cash_ledger.calendar.calculate_settlement_date(friday)
        
        # Should be Monday (T+1 business day)
        assert settlement_date.weekday() == 0  # Monday
        assert settlement_date > friday
    
    def test_end_of_day_settlement(self, cash_ledger):
        """Test end-of-day settlement processing."""
        # Execute a sell to create unsettled cash
        sell_transaction = cash_ledger.execute_sell(
            symbol="AAPL",
            quantity=10,
            price=Decimal('160.00'),
            trade_date=date(2024, 1, 15),
        )
        
        settlement_date = sell_transaction.settlement_date
        unsettled_amount = cash_ledger.state.unsettled_cash_by_date[settlement_date]
        
        # Process settlement
        settled_amount = cash_ledger.end_of_day_settlement(settlement_date)
        
        assert settled_amount == unsettled_amount
        assert settlement_date not in cash_ledger.state.unsettled_cash_by_date
        assert cash_ledger.state.settled_cash == Decimal('25000.00') + unsettled_amount
    
    def test_release_reserved_cash(self, cash_ledger):
        """Test releasing reserved cash."""
        # Execute a buy to reserve cash
        cash_ledger.execute_buy("AAPL", 10, Decimal('150.00'))
        
        initial_reserved = cash_ledger.state.reserved_cash
        initial_settled = cash_ledger.state.settled_cash
        
        # Release some reserved cash
        release_amount = Decimal('500.00')
        cash_ledger.release_reserved_cash(release_amount, "Order cancelled")
        
        assert cash_ledger.state.reserved_cash == initial_reserved - release_amount
        assert cash_ledger.state.settled_cash == initial_settled + release_amount
    
    def test_release_too_much_reserved_cash(self, cash_ledger):
        """Test releasing more cash than reserved."""
        cash_ledger.execute_buy("AAPL", 10, Decimal('150.00'))
        
        with pytest.raises(ValueError, match="Cannot release"):
            cash_ledger.release_reserved_cash(Decimal('2000.00'))
    
    def test_cash_summary(self, cash_ledger):
        """Test cash summary generation."""
        # Execute some transactions
        cash_ledger.execute_buy("AAPL", 10, Decimal('150.00'))
        cash_ledger.execute_sell("MSFT", 5, Decimal('200.00'))
        
        summary = cash_ledger.get_cash_summary()
        
        assert "settled_cash" in summary
        assert "reserved_cash" in summary
        assert "available_cash" in summary
        assert "unsettled_cash" in summary
        assert "total_cash" in summary
        assert "equity" in summary
        
        # Check that available cash = settled - reserved
        assert summary["available_cash"] == summary["settled_cash"] - summary["reserved_cash"]
    
    def test_unsettled_schedule(self, cash_ledger):
        """Test unsettled cash schedule."""
        # Execute a sell
        sell_transaction = cash_ledger.execute_sell(
            symbol="AAPL",
            quantity=10,
            price=Decimal('160.00'),
            trade_date=date(2024, 1, 15),
        )
        
        schedule = cash_ledger.get_unsettled_schedule()
        
        assert len(schedule) == 1
        settlement_date_str = sell_transaction.settlement_date.isoformat()
        assert settlement_date_str in schedule
        assert schedule[settlement_date_str] == 1600.0


if __name__ == "__main__":
    pytest.main([__file__])
