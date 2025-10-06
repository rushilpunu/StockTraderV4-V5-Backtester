"""Unit tests for bucket scheduler."""

import pytest
from datetime import date
from decimal import Decimal

from Traderv5.bucket_scheduler import BucketScheduler, BucketType
from Traderv5.configuration import CalendarSettings, SchedulerSettings


class TestBucketScheduler:
    """Test cases for BucketScheduler."""
    
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
    def scheduler_settings(self):
        """Create test scheduler settings."""
        return SchedulerSettings(
            bucket_a_days=("MONDAY", "WEDNESDAY", "FRIDAY"),
            bucket_b_days=("TUESDAY", "THURSDAY"),
            allocation_pct=0.5,
        )
    
    @pytest.fixture
    def bucket_scheduler(self, scheduler_settings, calendar_settings):
        """Create test bucket scheduler."""
        return BucketScheduler(scheduler_settings, calendar_settings)
    
    def test_initialization(self, bucket_scheduler):
        """Test bucket scheduler initialization."""
        assert bucket_scheduler.schedule.bucket_a_days == {"MONDAY", "WEDNESDAY", "FRIDAY"}
        assert bucket_scheduler.schedule.bucket_b_days == {"TUESDAY", "THURSDAY"}
        assert bucket_scheduler.schedule.allocation_pct == 0.5
    
    def test_get_active_bucket_monday(self, bucket_scheduler):
        """Test getting active bucket for Monday."""
        monday = date(2024, 1, 15)  # Monday
        active_bucket = bucket_scheduler.get_active_bucket(monday)
        assert active_bucket == BucketType.A
    
    def test_get_active_bucket_tuesday(self, bucket_scheduler):
        """Test getting active bucket for Tuesday."""
        tuesday = date(2024, 1, 16)  # Tuesday
        active_bucket = bucket_scheduler.get_active_bucket(tuesday)
        assert active_bucket == BucketType.B
    
    def test_get_active_bucket_weekend(self, bucket_scheduler):
        """Test getting active bucket for weekend."""
        saturday = date(2024, 1, 13)  # Saturday
        active_bucket = bucket_scheduler.get_active_bucket(saturday)
        assert active_bucket == BucketType.NONE
    
    def test_update_equity(self, bucket_scheduler):
        """Test updating equity allocation."""
        bucket_scheduler.update_equity(Decimal('20000.00'))
        
        bucket_a = bucket_scheduler.buckets[BucketType.A]
        bucket_b = bucket_scheduler.buckets[BucketType.B]
        
        # Each bucket should get 50% of equity
        assert bucket_a.allocated_amount == Decimal('10000.00')
        assert bucket_b.allocated_amount == Decimal('10000.00')
        assert bucket_a.available_amount == Decimal('10000.00')
        assert bucket_b.available_amount == Decimal('10000.00')
    
    def test_can_trade_sufficient_funds(self, bucket_scheduler):
        """Test can_trade with sufficient funds."""
        bucket_scheduler.update_equity(Decimal('20000.00'))
        
        monday = date(2024, 1, 15)  # Monday (Bucket A)
        can_trade, reason = bucket_scheduler.can_trade(Decimal('5000.00'), monday)
        
        assert can_trade is True
        assert "Trade allowed with bucket A" in reason
    
    def test_can_trade_insufficient_funds(self, bucket_scheduler):
        """Test can_trade with insufficient funds."""
        bucket_scheduler.update_equity(Decimal('20000.00'))
        
        monday = date(2024, 1, 15)  # Monday (Bucket A)
        can_trade, reason = bucket_scheduler.can_trade(Decimal('15000.00'), monday)
        
        assert can_trade is False
        assert "Insufficient bucket funds" in reason
    
    def test_can_trade_no_trading_day(self, bucket_scheduler):
        """Test can_trade on non-trading day."""
        saturday = date(2024, 1, 13)  # Saturday
        can_trade, reason = bucket_scheduler.can_trade(Decimal('1000.00'), saturday)
        
        assert can_trade is False
        assert "No trading allowed on this day" in reason
    
    def test_reserve_funds_success(self, bucket_scheduler):
        """Test successful fund reservation."""
        bucket_scheduler.update_equity(Decimal('20000.00'))
        
        monday = date(2024, 1, 15)  # Monday (Bucket A)
        success = bucket_scheduler.reserve_funds(Decimal('5000.00'), monday)
        
        assert success is True
        
        bucket_a = bucket_scheduler.buckets[BucketType.A]
        assert bucket_a.reserved_amount == Decimal('5000.00')
        assert bucket_a.free_amount == Decimal('5000.00')  # 10000 - 5000
    
    def test_reserve_funds_insufficient(self, bucket_scheduler):
        """Test fund reservation with insufficient funds."""
        bucket_scheduler.update_equity(Decimal('20000.00'))
        
        monday = date(2024, 1, 15)  # Monday (Bucket A)
        success = bucket_scheduler.reserve_funds(Decimal('15000.00'), monday)
        
        assert success is False
        
        bucket_a = bucket_scheduler.buckets[BucketType.A]
        assert bucket_a.reserved_amount == Decimal('0')
    
    def test_release_funds(self, bucket_scheduler):
        """Test releasing reserved funds."""
        bucket_scheduler.update_equity(Decimal('20000.00'))
        
        # Reserve funds first
        monday = date(2024, 1, 15)  # Monday (Bucket A)
        bucket_scheduler.reserve_funds(Decimal('5000.00'), monday)
        
        # Release funds
        bucket_scheduler.release_funds(Decimal('2000.00'), BucketType.A)
        
        bucket_a = bucket_scheduler.buckets[BucketType.A]
        assert bucket_a.reserved_amount == Decimal('3000.00')  # 5000 - 2000
        assert bucket_a.free_amount == Decimal('7000.00')  # 10000 - 3000
    
    def test_execute_trade(self, bucket_scheduler):
        """Test executing a trade."""
        bucket_scheduler.update_equity(Decimal('20000.00'))
        
        # Reserve funds first
        monday = date(2024, 1, 15)  # Monday (Bucket A)
        bucket_scheduler.reserve_funds(Decimal('5000.00'), monday)
        
        # Execute trade
        success = bucket_scheduler.execute_trade(Decimal('5000.00'), monday)
        
        assert success is True
        
        bucket_a = bucket_scheduler.buckets[BucketType.A]
        assert bucket_a.reserved_amount == Decimal('0')  # Consumed
        assert bucket_a.available_amount == Decimal('5000.00')  # 10000 - 5000
    
    def test_execute_trade_insufficient_reserved(self, bucket_scheduler):
        """Test executing trade with insufficient reserved funds."""
        bucket_scheduler.update_equity(Decimal('20000.00'))
        
        # Reserve less than we try to execute
        monday = date(2024, 1, 15)  # Monday (Bucket A)
        bucket_scheduler.reserve_funds(Decimal('3000.00'), monday)
        
        # Try to execute more than reserved
        success = bucket_scheduler.execute_trade(Decimal('5000.00'), monday)
        
        assert success is False
    
    def test_get_bucket_status(self, bucket_scheduler):
        """Test getting bucket status."""
        bucket_scheduler.update_equity(Decimal('20000.00'))
        
        status = bucket_scheduler.get_bucket_status()
        
        assert "A" in status
        assert "B" in status
        
        bucket_a_status = status["A"]
        assert bucket_a_status["allocated"] == 10000.0
        assert bucket_a_status["available"] == 10000.0
        assert bucket_a_status["reserved"] == 0.0
        assert bucket_a_status["free"] == 10000.0
    
    def test_get_active_bucket_info(self, bucket_scheduler):
        """Test getting active bucket information."""
        bucket_scheduler.update_equity(Decimal('20000.00'))
        
        monday = date(2024, 1, 15)  # Monday (Bucket A)
        info = bucket_scheduler.get_active_bucket_info(monday)
        
        assert info["bucket"] == "A"
        assert info["can_trade"] is True
        assert info["allocated"] == 10000.0
        assert info["available"] == 10000.0
    
    def test_get_active_bucket_info_weekend(self, bucket_scheduler):
        """Test getting active bucket information for weekend."""
        saturday = date(2024, 1, 13)  # Saturday
        info = bucket_scheduler.get_active_bucket_info(saturday)
        
        assert info["bucket"] == "NONE"
        assert info["can_trade"] is False
        assert "No trading allowed" in info["reason"]
    
    def test_get_trading_schedule(self, bucket_scheduler):
        """Test getting trading schedule."""
        start_date = date(2024, 1, 15)  # Monday
        end_date = date(2024, 1, 19)   # Friday
        
        schedule = bucket_scheduler.get_trading_schedule(start_date, end_date)
        
        # Should have 5 trading days
        assert len(schedule) == 5
        
        # Check specific days
        assert schedule["2024-01-15"] == "A"  # Monday
        assert schedule["2024-01-16"] == "B"  # Tuesday
        assert schedule["2024-01-17"] == "A"  # Wednesday
        assert schedule["2024-01-18"] == "B"  # Thursday
        assert schedule["2024-01-19"] == "A"  # Friday
    
    def test_reset_daily(self, bucket_scheduler):
        """Test daily reset functionality."""
        bucket_scheduler.update_equity(Decimal('20000.00'))
        
        # Reserve some funds
        monday = date(2024, 1, 15)  # Monday (Bucket A)
        bucket_scheduler.reserve_funds(Decimal('5000.00'), monday)
        
        # Reset daily
        bucket_scheduler.reset_daily()
        
        bucket_a = bucket_scheduler.buckets[BucketType.A]
        assert bucket_a.reserved_amount == Decimal('0')


if __name__ == "__main__":
    pytest.main([__file__])
