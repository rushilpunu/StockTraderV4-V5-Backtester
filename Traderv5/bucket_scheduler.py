"""Bucket scheduler for alternating trading days to manage cash account limitations."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Set

from Traderv5.calendars import TradingCalendar
from Traderv5.configuration import CalendarSettings, SchedulerSettings

_LOG = logging.getLogger("traderv5.bucket_scheduler")


class BucketType(Enum):
    """Trading bucket types."""
    A = "A"
    B = "B"
    NONE = "NONE"  # No trading allowed


@dataclass
class BucketAllocation:
    """Allocation of funds to a trading bucket."""
    bucket_type: BucketType
    allocated_amount: Decimal
    available_amount: Decimal
    reserved_amount: Decimal = Decimal('0')
    
    @property
    def free_amount(self) -> Decimal:
        """Amount available for new trades."""
        return self.available_amount - self.reserved_amount


@dataclass
class BucketSchedule:
    """Schedule of which bucket is active on which days."""
    bucket_a_days: Set[str]
    bucket_b_days: Set[str]
    allocation_pct: float
    
    def get_active_bucket(self, weekday: str) -> BucketType:
        """Get the active bucket for a given weekday."""
        weekday_upper = weekday.upper()
        if weekday_upper in self.bucket_a_days:
            return BucketType.A
        elif weekday_upper in self.bucket_b_days:
            return BucketType.B
        else:
            return BucketType.NONE


class BucketScheduler:
    """
    Manages alternating trading buckets to work around cash account limitations.
    
    The scheduler splits equity into two buckets (A and B) and alternates trading days
    between them. This ensures that while one bucket is settling (T+1), the other
    bucket can still trade, effectively allowing daily trading in a cash account.
    """
    
    def __init__(self, scheduler_settings: SchedulerSettings, calendar_settings: CalendarSettings):
        self.settings = scheduler_settings
        self.calendar = TradingCalendar(calendar_settings)
        
        # Create schedule
        self.schedule = BucketSchedule(
            bucket_a_days=set(day.upper() for day in scheduler_settings.bucket_a_days),
            bucket_b_days=set(day.upper() for day in scheduler_settings.bucket_b_days),
            allocation_pct=scheduler_settings.allocation_pct,
        )
        
        # Initialize bucket allocations
        self.buckets: Dict[BucketType, BucketAllocation] = {
            BucketType.A: BucketAllocation(
                bucket_type=BucketType.A,
                allocated_amount=Decimal('0'),
                available_amount=Decimal('0'),
            ),
            BucketType.B: BucketAllocation(
                bucket_type=BucketType.B,
                allocated_amount=Decimal('0'),
                available_amount=Decimal('0'),
            ),
        }
        
        _LOG.info(
            "Initialized bucket scheduler: A=%s, B=%s, allocation=%.1f%%",
            scheduler_settings.bucket_a_days,
            scheduler_settings.bucket_b_days,
            scheduler_settings.allocation_pct * 100
        )
    
    def update_equity(self, total_equity: Decimal) -> None:
        """
        Update bucket allocations based on total equity.
        
        Args:
            total_equity: Total account equity
        """
        # Calculate allocation per bucket
        allocation_per_bucket = total_equity * Decimal(str(self.settings.allocation_pct))
        
        # Update bucket allocations
        for bucket_type in [BucketType.A, BucketType.B]:
            bucket = self.buckets[bucket_type]
            bucket.allocated_amount = allocation_per_bucket
            bucket.available_amount = allocation_per_bucket - bucket.reserved_amount
        
        _LOG.info(
            "Updated equity: $%.2f total, $%.2f per bucket",
            total_equity,
            allocation_per_bucket
        )
    
    def get_active_bucket(self, target_date: Optional[date] = None) -> BucketType:
        """
        Get the active bucket for a given date.
        
        Args:
            target_date: Date to check (defaults to today)
            
        Returns:
            Active bucket type
        """
        if target_date is None:
            target_date = date.today()
        
        # Get weekday name
        weekday = target_date.strftime("%A").upper()
        return self.schedule.get_active_bucket(weekday)
    
    def can_trade(self, amount: Decimal, target_date: Optional[date] = None) -> tuple[bool, str]:
        """
        Check if a trade can be executed with the active bucket.
        
        Args:
            amount: Trade amount
            target_date: Date to check (defaults to today)
            
        Returns:
            Tuple of (can_trade, reason)
        """
        active_bucket = self.get_active_bucket(target_date)
        
        if active_bucket == BucketType.NONE:
            return False, "No trading allowed on this day"
        
        bucket = self.buckets[active_bucket]
        
        if amount <= 0:
            return False, "Trade amount must be positive"
        
        if amount > bucket.free_amount:
            return False, f"Insufficient bucket funds. Need ${amount}, have ${bucket.free_amount}"
        
        return True, f"Trade allowed with bucket {active_bucket.value}"
    
    def reserve_funds(self, amount: Decimal, target_date: Optional[date] = None) -> bool:
        """
        Reserve funds in the active bucket for a trade.
        
        Args:
            amount: Amount to reserve
            target_date: Date to check (defaults to today)
            
        Returns:
            True if reservation successful, False otherwise
        """
        active_bucket = self.get_active_bucket(target_date)
        
        if active_bucket == BucketType.NONE:
            return False
        
        bucket = self.buckets[active_bucket]
        
        if amount > bucket.free_amount:
            return False
        
        bucket.reserved_amount += amount
        _LOG.info(
            "Reserved $%.2f in bucket %s (free: $%.2f)",
            amount, active_bucket.value, bucket.free_amount
        )
        
        return True
    
    def release_funds(self, amount: Decimal, bucket_type: Optional[BucketType] = None) -> None:
        """
        Release reserved funds from a bucket.
        
        Args:
            amount: Amount to release
            bucket_type: Specific bucket to release from (defaults to active bucket)
        """
        if bucket_type is None:
            bucket_type = self.get_active_bucket()
        
        if bucket_type == BucketType.NONE:
            return
        
        bucket = self.buckets[bucket_type]
        
        if amount > bucket.reserved_amount:
            _LOG.warning(
                "Attempted to release $%.2f but only $%.2f reserved in bucket %s",
                amount, bucket.reserved_amount, bucket_type.value
            )
            amount = bucket.reserved_amount
        
        bucket.reserved_amount -= amount
        _LOG.info(
            "Released $%.2f from bucket %s (reserved: $%.2f)",
            amount, bucket_type.value, bucket.reserved_amount
        )
    
    def execute_trade(self, amount: Decimal, target_date: Optional[date] = None) -> bool:
        """
        Execute a trade by consuming reserved funds.
        
        Args:
            amount: Trade amount
            target_date: Date to check (defaults to today)
            
        Returns:
            True if trade executed successfully
        """
        active_bucket = self.get_active_bucket(target_date)
        
        if active_bucket == BucketType.NONE:
            return False
        
        bucket = self.buckets[active_bucket]
        
        if amount > bucket.reserved_amount:
            return False
        
        # Consume the reserved funds
        bucket.reserved_amount -= amount
        bucket.available_amount -= amount
        
        _LOG.info(
            "Executed trade: $%.2f from bucket %s (available: $%.2f)",
            amount, active_bucket.value, bucket.available_amount
        )
        
        return True
    
    def get_bucket_status(self) -> Dict[str, Dict[str, float]]:
        """Get current status of all buckets."""
        status = {}
        
        for bucket_type, bucket in self.buckets.items():
            status[bucket_type.value] = {
                "allocated": float(bucket.allocated_amount),
                "available": float(bucket.available_amount),
                "reserved": float(bucket.reserved_amount),
                "free": float(bucket.free_amount),
            }
        
        return status
    
    def get_active_bucket_info(self, target_date: Optional[date] = None) -> Dict[str, any]:
        """Get information about the active bucket for a given date."""
        active_bucket = self.get_active_bucket(target_date)
        
        if active_bucket == BucketType.NONE:
            return {
                "bucket": "NONE",
                "can_trade": False,
                "reason": "No trading allowed on this day",
            }
        
        bucket = self.buckets[active_bucket]
        
        return {
            "bucket": active_bucket.value,
            "can_trade": True,
            "allocated": float(bucket.allocated_amount),
            "available": float(bucket.available_amount),
            "reserved": float(bucket.reserved_amount),
            "free": float(bucket.free_amount),
        }
    
    def get_trading_schedule(self, start_date: date, end_date: date) -> Dict[str, str]:
        """
        Get the trading schedule for a date range.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            Dictionary mapping dates to active bucket
        """
        schedule = {}
        current_date = start_date
        
        while current_date <= end_date:
            if self.calendar.is_trading_day(current_date):
                active_bucket = self.get_active_bucket(current_date)
                schedule[current_date.isoformat()] = active_bucket.value
            current_date = self.calendar.next_trading_day(current_date)
        
        return schedule
    
    def reset_daily(self) -> None:
        """Reset daily state (called at start of each trading day)."""
        # Release any remaining reserved funds from previous day
        for bucket in self.buckets.values():
            if bucket.reserved_amount > 0:
                _LOG.info(
                    "Releasing $%.2f reserved funds from bucket %s",
                    bucket.reserved_amount, bucket.bucket_type.value
                )
                bucket.reserved_amount = Decimal('0')
        
        _LOG.info("Daily reset completed")


__all__ = [
    "BucketScheduler",
    "BucketType",
    "BucketAllocation",
    "BucketSchedule",
]
