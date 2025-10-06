"""Unit tests for decision router."""

import pytest
from datetime import date, datetime
from decimal import Decimal

from Traderv5.cash_ledger import CashLedger
from Traderv5.bucket_scheduler import BucketScheduler, BucketType
from Traderv5.decision_router import DecisionRouter, ModelPrediction, PositionInfo, TradeAction, TradeHorizon
from Traderv5.configuration import CalendarSettings, SchedulerSettings, RoutingSettings


class TestDecisionRouter:
    """Test cases for DecisionRouter."""
    
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
    def routing_settings(self):
        """Create test routing settings."""
        return RoutingSettings(
            swing_threshold=0.6,
            intraday_threshold=0.75,
            intraday_min_cash_pct=0.2,
            intraday_top_decile=0.9,
            hold_if_unsettled_needed=True,
        )
    
    @pytest.fixture
    def cash_ledger(self, calendar_settings):
        """Create test cash ledger."""
        return CashLedger(25000.0, calendar_settings)
    
    @pytest.fixture
    def bucket_scheduler(self, scheduler_settings, calendar_settings):
        """Create test bucket scheduler."""
        scheduler = BucketScheduler(scheduler_settings, calendar_settings)
        scheduler.update_equity(Decimal('25000.00'))
        return scheduler
    
    @pytest.fixture
    def decision_router(self, cash_ledger, bucket_scheduler, routing_settings):
        """Create test decision router."""
        return DecisionRouter(cash_ledger, bucket_scheduler, routing_settings)
    
    @pytest.fixture
    def sample_prediction(self):
        """Create sample model prediction."""
        return ModelPrediction(
            symbol="AAPL",
            swing_score=0.7,
            intraday_score=0.8,
            confidence=0.85,
            timestamp=datetime.utcnow(),
            features={"feature1": 0.5, "feature2": 0.3},
        )
    
    @pytest.fixture
    def sample_position(self):
        """Create sample position."""
        return PositionInfo(
            symbol="AAPL",
            quantity=10,
            avg_price=Decimal('150.00'),
            current_price=Decimal('155.00'),
            unrealized_pnl=Decimal('50.00'),
            days_held=5,
        )
    
    def test_route_buy_decision_swing_signal(self, decision_router, sample_prediction):
        """Test routing buy decision with swing signal."""
        current_price = Decimal('150.00')
        current_date = date(2024, 1, 15)  # Monday (Bucket A)
        
        decision = decision_router._route_buy_decision(sample_prediction, current_price, current_date)
        
        assert decision.action == TradeAction.BUY
        assert decision.symbol == "AAPL"
        assert decision.horizon == TradeHorizon.SWING
        assert decision.bucket_type == BucketType.A
        assert decision.quantity > 0
        assert decision.price == current_price
    
    def test_route_buy_decision_intraday_signal(self, decision_router):
        """Test routing buy decision with intraday signal."""
        prediction = ModelPrediction(
            symbol="AAPL",
            swing_score=0.5,  # Below threshold
            intraday_score=0.8,  # Above threshold
            confidence=0.85,
            timestamp=datetime.utcnow(),
            features={},
        )
        
        current_price = Decimal('150.00')
        current_date = date(2024, 1, 15)  # Monday (Bucket A)
        
        decision = decision_router._route_buy_decision(prediction, current_price, current_date)
        
        assert decision.action == TradeAction.BUY
        assert decision.horizon == TradeHorizon.INTRADAY
    
    def test_route_buy_decision_below_threshold(self, decision_router):
        """Test routing buy decision with scores below threshold."""
        prediction = ModelPrediction(
            symbol="AAPL",
            swing_score=0.4,  # Below threshold
            intraday_score=0.5,  # Below threshold
            confidence=0.85,
            timestamp=datetime.utcnow(),
            features={},
        )
        
        current_price = Decimal('150.00')
        current_date = date(2024, 1, 15)  # Monday (Bucket A)
        
        decision = decision_router._route_buy_decision(prediction, current_price, current_date)
        
        assert decision.action == TradeAction.SKIP
        assert "Model scores below threshold" in decision.reason
    
    def test_route_buy_decision_no_trading_day(self, decision_router, sample_prediction):
        """Test routing buy decision on non-trading day."""
        current_price = Decimal('150.00')
        saturday = date(2024, 1, 13)  # Saturday
        
        decision = decision_router._route_buy_decision(sample_prediction, current_price, saturday)
        
        assert decision.action == TradeAction.SKIP
        assert "No trading allowed on this day" in decision.reason
        assert decision.bucket_type == BucketType.NONE
    
    def test_route_buy_decision_insufficient_cash(self, decision_router, sample_prediction):
        """Test routing buy decision with insufficient cash."""
        # Use up most of the cash
        decision_router.cash_ledger.execute_buy("MSFT", 100, Decimal('200.00'))
        
        current_price = Decimal('150.00')
        current_date = date(2024, 1, 15)  # Monday (Bucket A)
        
        decision = decision_router._route_buy_decision(sample_prediction, current_price, current_date)
        
        assert decision.action == TradeAction.SKIP
        assert "Insufficient cash" in decision.reason
    
    def test_route_buy_decision_insufficient_bucket_funds(self, decision_router, sample_prediction):
        """Test routing buy decision with insufficient bucket funds."""
        # Reserve most of bucket A funds
        decision_router.bucket_scheduler.reserve_funds(Decimal('12000.00'), date(2024, 1, 15))
        
        current_price = Decimal('150.00')
        current_date = date(2024, 1, 15)  # Monday (Bucket A)
        
        decision = decision_router._route_buy_decision(sample_prediction, current_price, current_date)
        
        assert decision.action == TradeAction.SKIP
        assert "Bucket constraint" in decision.reason
    
    def test_route_sell_decision_sell_signal(self, decision_router, sample_position):
        """Test routing sell decision with sell signal."""
        prediction = ModelPrediction(
            symbol="AAPL",
            swing_score=-0.7,  # Negative (sell signal)
            intraday_score=-0.8,  # Negative (sell signal)
            confidence=0.85,
            timestamp=datetime.utcnow(),
            features={},
        )
        
        current_price = Decimal('155.00')
        current_date = date(2024, 1, 15)
        
        decision = decision_router._route_sell_decision(prediction, sample_position, current_price, current_date)
        
        assert decision.action == TradeAction.SELL
        assert decision.symbol == "AAPL"
        assert decision.quantity == sample_position.quantity
        assert "Sell signal" in decision.reason
    
    def test_route_sell_decision_hold(self, decision_router, sample_position):
        """Test routing sell decision with no sell signal."""
        prediction = ModelPrediction(
            symbol="AAPL",
            swing_score=0.3,  # Positive (no sell signal)
            intraday_score=0.4,  # Positive (no sell signal)
            confidence=0.85,
            timestamp=datetime.utcnow(),
            features={},
        )
        
        current_price = Decimal('155.00')
        current_date = date(2024, 1, 15)
        
        decision = decision_router._route_sell_decision(prediction, sample_position, current_price, current_date)
        
        assert decision.action == TradeAction.HOLD
        assert "No sell signal" in decision.reason
    
    def test_route_decision_with_position(self, decision_router, sample_prediction, sample_position):
        """Test routing decision when position exists."""
        current_price = Decimal('155.00')
        current_date = date(2024, 1, 15)
        current_positions = [sample_position]
        
        decision = decision_router.route_decision(sample_prediction, current_positions, current_price, current_date)
        
        # Should route to sell decision since position exists
        assert decision.action in [TradeAction.SELL, TradeAction.HOLD]
    
    def test_route_decision_without_position(self, decision_router, sample_prediction):
        """Test routing decision when no position exists."""
        current_price = Decimal('150.00')
        current_date = date(2024, 1, 15)  # Monday (Bucket A)
        current_positions = []
        
        decision = decision_router.route_decision(sample_prediction, current_positions, current_price, current_date)
        
        # Should route to buy decision since no position exists
        assert decision.action == TradeAction.BUY
    
    def test_calculate_risk_amount(self, decision_router):
        """Test risk amount calculation."""
        # Test with different scores and horizons
        swing_risk = decision_router._calculate_risk_amount(0.7, TradeHorizon.SWING)
        intraday_risk = decision_router._calculate_risk_amount(0.8, TradeHorizon.INTRADAY)
        
        assert swing_risk > 0
        assert intraday_risk > 0
        assert intraday_risk < swing_risk  # Intraday should have lower risk
    
    def test_calculate_quantity(self, decision_router):
        """Test quantity calculation."""
        risk_amount = Decimal('500.00')
        price = Decimal('150.00')
        
        quantity = decision_router._calculate_quantity(risk_amount, price)
        
        assert quantity > 0
        assert isinstance(quantity, int)
    
    def test_calculate_stop_take_profit(self, decision_router):
        """Test stop loss and take profit calculation."""
        entry_price = Decimal('150.00')
        risk_amount = Decimal('500.00')
        quantity = 10
        
        stop_loss, take_profit = decision_router._calculate_stop_take_profit(entry_price, risk_amount, quantity)
        
        assert stop_loss < entry_price
        assert take_profit > entry_price
        assert stop_loss > 0
        assert take_profit > 0


if __name__ == "__main__":
    pytest.main([__file__])
