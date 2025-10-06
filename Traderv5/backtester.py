"""Cash account mode backtester with settlement tracking and violation prevention."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from Traderv5.bucket_scheduler import BucketScheduler, BucketType
from Traderv5.calendars import TradingCalendar
from Traderv5.cash_ledger import CashLedger, CashTransaction
from Traderv5.configuration import TradingParameters
from Traderv5.decision_router import DecisionRouter, ModelPrediction, PositionInfo, TradeDecision, TradeAction
from Traderv5.execution_manager import ExecutionManager, ExecutionResult
from Traderv5.risk_manager import RiskManager, RiskMetrics

_LOG = logging.getLogger("traderv5.backtester")


@dataclass
class BacktestPosition:
    """Position in the backtest."""
    symbol: str
    quantity: int
    entry_price: Decimal
    entry_date: date
    current_price: Decimal
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    risk_metrics: Optional[RiskMetrics] = None
    
    @property
    def market_value(self) -> Decimal:
        return Decimal(str(self.quantity)) * self.current_price
    
    @property
    def unrealized_pnl(self) -> Decimal:
        return (self.current_price - self.entry_price) * Decimal(str(self.quantity))
    
    @property
    def unrealized_pnl_pct(self) -> Decimal:
        if self.entry_price <= 0:
            return Decimal('0')
        return (self.current_price - self.entry_price) / self.entry_price * Decimal('100')


@dataclass
class BacktestTrade:
    """Completed trade in the backtest."""
    symbol: str
    side: str  # "BUY" or "SELL"
    quantity: int
    entry_price: Decimal
    exit_price: Optional[Decimal]
    entry_date: date
    exit_date: Optional[date]
    pnl: Decimal
    pnl_pct: Decimal
    holding_days: int
    reason: str
    risk_amount: Decimal
    bucket_type: Optional[BucketType]
    
    @property
    def is_complete(self) -> bool:
        return self.exit_price is not None and self.exit_date is not None


@dataclass
class BacktestViolation:
    """Trading violation detected during backtest."""
    violation_type: str
    date: date
    symbol: str
    description: str
    severity: str  # "WARNING", "ERROR", "CRITICAL"


@dataclass
class BacktestResult:
    """Results of the backtest."""
    start_date: date
    end_date: date
    initial_equity: Decimal
    final_equity: Decimal
    total_return: Decimal
    total_return_pct: Decimal
    max_drawdown: Decimal
    max_drawdown_pct: Decimal
    sharpe_ratio: Optional[float]
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: Decimal
    avg_loss: Decimal
    profit_factor: Optional[float]
    violations: List[BacktestViolation]
    equity_curve: pd.DataFrame
    trades: List[BacktestTrade]
    daily_returns: List[Decimal]
    
    @property
    def is_successful(self) -> bool:
        """Check if backtest was successful (no critical violations)."""
        critical_violations = [v for v in self.violations if v.severity == "CRITICAL"]
        return len(critical_violations) == 0


class CashAccountBacktester:
    """
    Backtester specifically designed for cash accounts with settlement tracking.
    
    Features:
    - Tracks settled vs unsettled cash
    - Prevents good faith violations
    - Uses bucket scheduler for alternating trading days
    - Simulates T+1 settlement
    - Tracks all violations and compliance issues
    """
    
    def __init__(self, config: TradingParameters):
        self.config = config
        self.calendar = TradingCalendar(config.calendar)
        self.cash_ledger = CashLedger(config.account.starting_equity, config.calendar)
        self.bucket_scheduler = BucketScheduler(config.scheduler, config.calendar)
        self.risk_manager = RiskManager(config.risk)
        self.execution_manager = ExecutionManager(self.cash_ledger, self.risk_manager)
        
        # Backtest state
        self.positions: Dict[str, BacktestPosition] = {}
        self.completed_trades: List[BacktestTrade] = []
        self.violations: List[BacktestViolation] = []
        self.equity_curve: List[Tuple[date, Decimal]] = []
        self.daily_returns: List[Decimal] = []
        
        # Initialize bucket scheduler with starting equity
        self.bucket_scheduler.update_equity(Decimal(str(config.account.starting_equity)))
        
        _LOG.info("Initialized cash account backtester with $%.2f starting equity", config.account.starting_equity)
    
    def run_backtest(
        self,
        price_data: Dict[str, pd.DataFrame],
        predictions: Dict[str, List[ModelPrediction]],
        start_date: date,
        end_date: date,
    ) -> BacktestResult:
        """
        Run the backtest over the specified date range.
        
        Args:
            price_data: Dictionary of symbol -> DataFrame with OHLC data
            predictions: Dictionary of symbol -> List of ModelPredictions
            start_date: Start date for backtest
            end_date: End date for backtest
            
        Returns:
            BacktestResult with all metrics and violations
        """
        _LOG.info("Starting backtest from %s to %s", start_date, end_date)
        
        # Initialize equity curve
        self.equity_curve = [(start_date, Decimal(str(self.config.account.starting_equity)))]
        
        # Get trading days
        trading_days = self.calendar.trading_days(start_date, end_date)
        
        for current_date in trading_days:
            try:
                self._process_trading_day(current_date, price_data, predictions)
            except Exception as e:
                _LOG.error("Error processing trading day %s: %s", current_date, e)
                self._add_violation(
                    "SYSTEM_ERROR",
                    current_date,
                    "",
                    f"System error: {str(e)}",
                    "CRITICAL"
                )
        
        # Calculate final results
        result = self._calculate_results(start_date, end_date)
        
        _LOG.info(
            "Backtest completed: %.2f%% return, %d trades, %d violations",
            result.total_return_pct, result.total_trades, len(result.violations)
        )
        
        return result
    
    def _process_trading_day(
        self,
        current_date: date,
        price_data: Dict[str, pd.DataFrame],
        predictions: Dict[str, List[ModelPrediction]],
    ) -> None:
        """Process a single trading day."""
        _LOG.debug("Processing trading day: %s", current_date)
        
        # End of day settlement
        self.cash_ledger.end_of_day_settlement(current_date)
        
        # Reset bucket scheduler for new day
        self.bucket_scheduler.reset_daily()
        
        # Update current equity
        current_equity = self._calculate_current_equity(current_date, price_data)
        self.cash_ledger.update_equity(current_equity)
        self.bucket_scheduler.update_equity(current_equity)
        self.risk_manager.update_account_state(
            current_equity,
            self.cash_ledger.state.available_cash,
            self._get_current_risk_metrics(),
        )
        
        # Process predictions for this date
        for symbol, symbol_predictions in predictions.items():
            if symbol not in price_data:
                continue
            
            # Get prediction for current date
            prediction = self._get_prediction_for_date(symbol_predictions, current_date)
            if prediction is None:
                continue
            
            # Get current price
            current_price = self._get_price_for_date(price_data[symbol], current_date)
            if current_price is None:
                continue
            
            # Process the prediction
            self._process_prediction(prediction, current_price, current_date, price_data[symbol])
        
        # Check for stop losses and take profits
        self._check_exit_signals(current_date, price_data)
        
        # Record equity for the day
        self.equity_curve.append((current_date, current_equity))
    
    def _process_prediction(
        self,
        prediction: ModelPrediction,
        current_price: Decimal,
        current_date: date,
        price_data: pd.DataFrame,
    ) -> None:
        """Process a model prediction and execute trades if appropriate."""
        # Get current positions
        current_positions = self._get_current_positions()
        
        # Create decision router (simplified - in practice, you'd reuse the same instance)
        router = DecisionRouter(
            self.cash_ledger,
            self.bucket_scheduler,
            self.config.routing,
        )
        
        # Get decision
        decision = router.route_decision(prediction, current_positions, current_price, current_date)
        
        # Execute if needed
        if decision.action == TradeAction.BUY:
            self._execute_buy_decision(decision, current_price, current_date, price_data)
        elif decision.action == TradeAction.SELL:
            self._execute_sell_decision(decision, current_price, current_date)
    
    def _execute_buy_decision(
        self,
        decision: TradeDecision,
        current_price: Decimal,
        current_date: date,
        price_data: pd.DataFrame,
    ) -> None:
        """Execute a buy decision."""
        try:
            # Calculate ATR for risk management
            atr = self.risk_manager.calculate_atr(price_data)
            
            # Execute the trade
            execution_result = self.execution_manager.execute_trade_decision(decision, atr, current_price)
            
            if execution_result.success:
                # Create position
                position = BacktestPosition(
                    symbol=decision.symbol,
                    quantity=decision.quantity,
                    entry_price=current_price,
                    entry_date=current_date,
                    current_price=current_price,
                    stop_loss=decision.stop_loss,
                    take_profit=decision.take_profit,
                    risk_metrics=execution_result.risk_metrics,
                )
                
                self.positions[decision.symbol] = position
                
                _LOG.info(
                    "Opened position: %s %d shares @ $%.2f (bucket: %s)",
                    decision.symbol, decision.quantity, current_price,
                    decision.bucket_type.value if decision.bucket_type else "NONE"
                )
            else:
                self._add_violation(
                    "EXECUTION_FAILED",
                    current_date,
                    decision.symbol,
                    f"Failed to execute buy: {execution_result.message}",
                    "WARNING"
                )
                
        except Exception as e:
            self._add_violation(
                "EXECUTION_ERROR",
                current_date,
                decision.symbol,
                f"Error executing buy: {str(e)}",
                "ERROR"
            )
    
    def _execute_sell_decision(
        self,
        decision: TradeDecision,
        current_price: Decimal,
        current_date: date,
    ) -> None:
        """Execute a sell decision."""
        try:
            if decision.symbol not in self.positions:
                self._add_violation(
                    "NO_POSITION",
                    current_date,
                    decision.symbol,
                    "Attempted to sell without position",
                    "ERROR"
                )
                return
            
            position = self.positions[decision.symbol]
            
            # Execute the trade
            execution_result = self.execution_manager.execute_trade_decision(decision, Decimal('0'), current_price)
            
            if execution_result.success:
                # Create completed trade
                trade = BacktestTrade(
                    symbol=decision.symbol,
                    side="SELL",
                    quantity=position.quantity,
                    entry_price=position.entry_price,
                    exit_price=current_price,
                    entry_date=position.entry_date,
                    exit_date=current_date,
                    pnl=position.unrealized_pnl,
                    pnl_pct=position.unrealized_pnl_pct,
                    holding_days=(current_date - position.entry_date).days,
                    reason=decision.reason,
                    risk_amount=position.risk_metrics.risk_amount if position.risk_metrics else Decimal('0'),
                    bucket_type=decision.bucket_type,
                )
                
                self.completed_trades.append(trade)
                del self.positions[decision.symbol]
                
                _LOG.info(
                    "Closed position: %s %d shares @ $%.2f, P&L: $%.2f (%.2f%%)",
                    decision.symbol, position.quantity, current_price,
                    trade.pnl, trade.pnl_pct
                )
            else:
                self._add_violation(
                    "EXECUTION_FAILED",
                    current_date,
                    decision.symbol,
                    f"Failed to execute sell: {execution_result.message}",
                    "WARNING"
                )
                
        except Exception as e:
            self._add_violation(
                "EXECUTION_ERROR",
                current_date,
                decision.symbol,
                f"Error executing sell: {str(e)}",
                "ERROR"
            )
    
    def _check_exit_signals(self, current_date: date, price_data: Dict[str, pd.DataFrame]) -> None:
        """Check for stop loss and take profit signals."""
        for symbol, position in list(self.positions.items()):
            if symbol not in price_data:
                continue
            
            current_price = self._get_price_for_date(price_data[symbol], current_date)
            if current_price is None:
                continue
            
            # Update position price
            position.current_price = current_price
            
            # Check stop loss
            if position.stop_loss and current_price <= position.stop_loss:
                self._execute_sell_decision(
                    TradeDecision(
                        action=TradeAction.SELL,
                        symbol=symbol,
                        quantity=position.quantity,
                        price=current_price,
                        horizon=None,
                        stop_loss=None,
                        take_profit=None,
                        risk_amount=Decimal('0'),
                        reason="Stop loss triggered",
                        timestamp=datetime.utcnow(),
                        model_score=0.0,
                        bucket_type=None,
                    ),
                    current_price,
                    current_date,
                )
            
            # Check take profit
            elif position.take_profit and current_price >= position.take_profit:
                self._execute_sell_decision(
                    TradeDecision(
                        action=TradeAction.SELL,
                        symbol=symbol,
                        quantity=position.quantity,
                        price=current_price,
                        horizon=None,
                        stop_loss=None,
                        take_profit=None,
                        risk_amount=Decimal('0'),
                        reason="Take profit triggered",
                        timestamp=datetime.utcnow(),
                        model_score=0.0,
                        bucket_type=None,
                    ),
                    current_price,
                    current_date,
                )
    
    def _calculate_current_equity(self, current_date: date, price_data: Dict[str, pd.DataFrame]) -> Decimal:
        """Calculate current equity including positions."""
        cash_value = self.cash_ledger.state.total_cash
        position_value = Decimal('0')
        
        for symbol, position in self.positions.items():
            if symbol in price_data:
                current_price = self._get_price_for_date(price_data[symbol], current_date)
                if current_price is not None:
                    position_value += current_price * Decimal(str(position.quantity))
        
        return cash_value + position_value
    
    def _get_current_positions(self) -> List[PositionInfo]:
        """Get current positions in the format expected by DecisionRouter."""
        positions = []
        for symbol, position in self.positions.items():
            positions.append(PositionInfo(
                symbol=symbol,
                quantity=position.quantity,
                avg_price=position.entry_price,
                current_price=position.current_price,
                unrealized_pnl=position.unrealized_pnl,
                days_held=(date.today() - position.entry_date).days,
            ))
        return positions
    
    def _get_current_risk_metrics(self) -> List[RiskMetrics]:
        """Get current risk metrics for all positions."""
        return [
            position.risk_metrics for position in self.positions.values()
            if position.risk_metrics is not None
        ]
    
    def _get_prediction_for_date(
        self,
        predictions: List[ModelPrediction],
        target_date: date,
    ) -> Optional[ModelPrediction]:
        """Get prediction for a specific date."""
        # Prefer an exact match; otherwise return the latest prediction
        # prior to the target date (useful when intraday times vary).
        exact: Optional[ModelPrediction] = None
        latest_prior: Optional[ModelPrediction] = None
        for prediction in predictions:
            pred_day = prediction.timestamp.date()
            if pred_day == target_date:
                exact = prediction
                break
            if pred_day < target_date:
                if latest_prior is None or latest_prior.timestamp < prediction.timestamp:
                    latest_prior = prediction
        return exact or latest_prior
    
    def _get_price_for_date(self, price_data: pd.DataFrame, target_date: date) -> Optional[Decimal]:
        """Get price for a specific date."""
        try:
            if price_data.empty:
                return None
            # Normalize index to dates for matching
            index_dates = pd.to_datetime(price_data.index).date
            # Try exact match first
            matches = index_dates == target_date
            if matches.any():
                first_idx = np.nonzero(matches)[0][0]
                return Decimal(str(price_data.iloc[first_idx]['close']))
            # Otherwise, get the last available prior or equal bar
            prior_mask = index_dates <= target_date
            if prior_mask.any():
                last_idx = np.nonzero(prior_mask)[0][-1]
                return Decimal(str(price_data.iloc[last_idx]['close']))
            return None
        except Exception:
            return None
    
    def _add_violation(
        self,
        violation_type: str,
        date: date,
        symbol: str,
        description: str,
        severity: str,
    ) -> None:
        """Add a violation to the backtest results."""
        violation = BacktestViolation(
            violation_type=violation_type,
            date=date,
            symbol=symbol,
            description=description,
            severity=severity,
        )
        self.violations.append(violation)
        
        _LOG.warning("Violation [%s]: %s - %s", severity, violation_type, description)
    
    def _calculate_results(self, start_date: date, end_date: date) -> BacktestResult:
        """Calculate final backtest results."""
        initial_equity = Decimal(str(self.config.account.starting_equity))
        final_equity = self.equity_curve[-1][1] if self.equity_curve else initial_equity
        
        total_return = final_equity - initial_equity
        total_return_pct = (total_return / initial_equity) * Decimal('100')
        
        # Calculate drawdown
        equity_values = [equity for _, equity in self.equity_curve]
        peak_equity = initial_equity
        max_drawdown = Decimal('0')
        
        for equity in equity_values:
            if equity > peak_equity:
                peak_equity = equity
            drawdown = peak_equity - equity
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        max_drawdown_pct = (max_drawdown / initial_equity) * Decimal('100')
        
        # Calculate trade statistics
        winning_trades = [t for t in self.completed_trades if t.pnl > 0]
        losing_trades = [t for t in self.completed_trades if t.pnl < 0]
        
        win_rate = len(winning_trades) / len(self.completed_trades) if self.completed_trades else 0.0
        avg_win = sum(t.pnl for t in winning_trades) / len(winning_trades) if winning_trades else Decimal('0')
        avg_loss = sum(t.pnl for t in losing_trades) / len(losing_trades) if losing_trades else Decimal('0')
        
        profit_factor = None
        if losing_trades and avg_loss != 0:
            total_wins = sum(t.pnl for t in winning_trades)
            total_losses = abs(sum(t.pnl for t in losing_trades))
            profit_factor = float(total_wins / total_losses) if total_losses > 0 else None
        
        # Calculate Sharpe ratio (simplified)
        daily_returns = []
        for i in range(1, len(self.equity_curve)):
            prev_equity = self.equity_curve[i-1][1]
            curr_equity = self.equity_curve[i][1]
            daily_return = (curr_equity - prev_equity) / prev_equity
            daily_returns.append(daily_return)
        
        sharpe_ratio = None
        if daily_returns:
            returns_array = np.array([float(r) for r in daily_returns])
            if len(returns_array) > 1 and returns_array.std() > 0:
                sharpe_ratio = float(returns_array.mean() / returns_array.std() * np.sqrt(252))
        
        # Create equity curve DataFrame
        equity_df = pd.DataFrame(self.equity_curve, columns=['date', 'equity'])
        equity_df.set_index('date', inplace=True)
        
        return BacktestResult(
            start_date=start_date,
            end_date=end_date,
            initial_equity=initial_equity,
            final_equity=final_equity,
            total_return=total_return,
            total_return_pct=total_return_pct,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            sharpe_ratio=sharpe_ratio,
            total_trades=len(self.completed_trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            violations=self.violations,
            equity_curve=equity_df,
            trades=self.completed_trades,
            daily_returns=daily_returns,
        )


__all__ = [
    "CashAccountBacktester",
    "BacktestResult",
    "BacktestPosition",
    "BacktestTrade",
    "BacktestViolation",
]
