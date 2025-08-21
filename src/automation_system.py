"""Automation framework and monitoring system."""

import asyncio
import schedule
import time
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import json
import os
from loguru import logger

from .config import config
from .gdelt_client import GDELTClient
from .data_processor import DataProcessor
from .volatility_analyzer import VolatilityAnalyzer
from .alert_system import AlertSystem
from .trading_engine import TradingEngine
from .alpaca_client import AlpacaClient


@dataclass
class SystemStatus:
    """Container for system status information."""
    is_running: bool = False
    last_update: Optional[datetime] = None
    total_cycles: int = 0
    successful_cycles: int = 0
    failed_cycles: int = 0
    active_alerts: int = 0
    open_positions: int = 0
    daily_pnl: float = 0.0
    system_health: str = "unknown"  # healthy, warning, error
    errors: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CycleMetrics:
    """Metrics for a single automation cycle."""
    cycle_id: int
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    tickers_processed: int = 0
    alerts_generated: int = 0
    trades_executed: int = 0
    errors: List[str] = field(default_factory=list)
    success: bool = False


class AutomationSystem:
    """Comprehensive automation framework for the trading system."""
    
    def __init__(self):
        # Core components
        self.gdelt_client: Optional[GDELTClient] = None
        self.data_processor = DataProcessor()
        self.volatility_analyzer = VolatilityAnalyzer()
        self.alert_system = AlertSystem()
        self.trading_engine = TradingEngine()
        self.alpaca_client: Optional[AlpacaClient] = None
        
        # Automation state
        self.status = SystemStatus()
        self.is_running = False
        self.should_stop = False
        self.current_cycle: Optional[CycleMetrics] = None
        self.cycle_history: List[CycleMetrics] = []
        
        # Scheduling

        self.scheduled_tasks: List[Callable] = []
        
        # Monitoring
        self.health_checks: Dict[str, Callable] = {}
        self.performance_history: List[Dict[str, Any]] = []
        
        # Error handling
        self.max_consecutive_errors = 5
        self.consecutive_errors = 0
        self.last_error_time: Optional[datetime] = None
        
        self._initialize_health_checks()
        self._initialize_scheduled_tasks()
    
    def _initialize_health_checks(self):
        """Initialize health check functions."""
        self.health_checks = {
            "gdelt_api": self._check_gdelt_api_health,
            "alpaca_api": self._check_alpaca_api_health,
            "data_processor": self._check_data_processor_health,
            "trading_engine": self._check_trading_engine_health,
            "disk_space": self._check_disk_space,
            "memory_usage": self._check_memory_usage
        }
    
    def _initialize_scheduled_tasks(self):
        """Initialize scheduled tasks."""
        # Main data collection and analysis cycle
        schedule.every(config.analysis_interval_minutes).minutes.do(
            self._schedule_analysis_cycle
        )
        
        # Portfolio and order monitoring
        schedule.every(2).minutes.do(self._schedule_portfolio_monitoring)
        
        # System health checks
        schedule.every(10).minutes.do(self._schedule_health_check)
        
        # Daily reset and cleanup
        schedule.every().day.at("00:01").do(self._schedule_daily_reset)
        
        # Market hours check
        schedule.every(30).minutes.do(self._schedule_market_hours_check)
        
        # Performance logging
        schedule.every().hour.do(self._schedule_performance_logging)
    
    async def start(self, run_once: bool = False):
        """Start the automation system.
        
        Args:
            run_once: If True, run a single analysis cycle and exit.
        """
        logger.info("Starting automation system...")
        
        try:
            # Initialize clients
            await self._initialize_clients()
            
            # Perform initial health check
            health_status = await self._perform_health_check()
            if health_status["overall_health"] == "error":
                logger.warning("Initial health check failed - continuing with warnings")
                # Continue anyway for demonstration purposes
                # return False
            
            # Start the main automation loop
            self.is_running = True
            self.should_stop = False
            self.status.is_running = True
            self.status.system_health = "healthy"
            
            logger.info("Automation system started successfully")

            # Kick off an immediate analysis cycle so users don't wait for the first schedule interval
            try:
                await self._run_analysis_cycle()
            except Exception as e:
                logger.error(f"Immediate analysis cycle failed: {e}")
                self._handle_error(str(e))
            
            # If run-once mode is enabled, exit after the immediate cycle
            if run_once:
                logger.info("Run-once mode enabled - exiting after single analysis cycle")
                return True

            # Start the main loop
            await self._run_automation_loop()
            
        except Exception as e:
            logger.error(f"Failed to start automation system: {e}")
            self.status.system_health = "error"
            self.status.errors.append(f"Startup failed: {str(e)}")
            return False
    async def stop(self):
        """Stop the automation system."""
        logger.info("Stopping automation system...")
        
        self.is_running = False
        self.status.is_running = False
        
        # Close clients
        if self.gdelt_client:
            await self.gdelt_client.__aexit__(None, None, None)
        

        
        logger.info("Automation system stopped")
    
    async def _initialize_clients(self):
        """Initialize all API clients."""
        try:
            # Initialize GDELT client
            self.gdelt_client = GDELTClient()
            await self.gdelt_client.__aenter__()
            
            # Initialize Alpaca client
            self.alpaca_client = AlpacaClient()
            
            # Test connections
            account_info = await self.alpaca_client.get_account_info()
            if not account_info:
                raise Exception("Failed to connect to Alpaca API")
            
            logger.info("All clients initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize clients: {e}")
            raise
    
    async def _run_automation_loop(self):
        """Main automation loop."""
        logger.info("Starting main automation loop")
        
        while not self.should_stop:
            try:
                # Run scheduled tasks
                schedule.run_pending()
                
                # Short sleep to prevent busy waiting
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in automation loop: {e}")
                self._handle_error(str(e))
                
                # If too many consecutive errors, stop the system
                if self.consecutive_errors >= self.max_consecutive_errors:
                    logger.critical("Too many consecutive errors - stopping system")
                    await self.stop()
                    break
                
                # Wait before retrying
                await asyncio.sleep(30)
    
    def _schedule_analysis_cycle(self):
        """Schedule a complete analysis cycle."""
        if not self.is_running:
            return
        
        # Create a task for the analysis cycle
        asyncio.create_task(self._run_analysis_cycle())
    
    async def _run_analysis_cycle(self):
        """Run a complete analysis cycle for all tickers."""
        cycle_id = self.status.total_cycles + 1
        cycle = CycleMetrics(
            cycle_id=cycle_id,
            start_time=datetime.utcnow()
        )
        
        self.current_cycle = cycle
        
        logger.info(f"Starting analysis cycle #{cycle_id}")
        
        try:
            # Check market hours but continue analysis to allow aggressive mode
            market_hours = await self.alpaca_client.get_market_hours()
            if not market_hours.get("is_open", False):
                logger.info("Market is closed - continuing analysis in aggressive mode")
            
            # Process each ticker
            for ticker in config.stock_tickers:
                # Skip very short tickers that GDELT will reject
                if len(ticker) < 3:
                    logger.info(f"Skipping {ticker} - ticker too short for GDELT API")
                    continue
                    
                try:
                    await self._process_ticker(ticker, cycle)
                    cycle.tickers_processed += 1
                    
                except Exception as e:
                    error_msg = f"Error processing {ticker}: {str(e)}"
                    logger.error(error_msg)
                    cycle.errors.append(error_msg)
                    continue
            
            # Update portfolio positions
            await self._update_portfolio_positions()
            
            # Check for trade execution opportunities (immediately process fresh alerts first)
            await self._process_pending_alerts()
            
            # Immediate exit enforcement within the cycle (useful for --once and tighter loops)
            try:
                await self._update_portfolio_positions()
                await self._refresh_position_prices()
                await self._enforce_position_exits()
                await self._enforce_exits_by_pnl()
            except Exception as e:
                logger.error(f"In-cycle exit enforcement failed: {e}")

            # Aggressive mode: ensure minimum trades per cycle
            try:
                if (config.aggressive_mode_enabled and self.current_cycle and 
                    self.current_cycle.trades_executed < max(0, config.min_trades_per_cycle)):
                    logger.info(
                        f"Aggressive mode active. Trades this cycle: {self.current_cycle.trades_executed}. "
                        f"Target minimum: {config.min_trades_per_cycle}. Forcing a trade."
                    )
                    await self._force_trade_for_cycle()
            except Exception as e:
                logger.error(f"Aggressive mode force-trade failed: {e}")
            
            cycle.success = len(cycle.errors) < len(config.stock_tickers) / 2
            
        except Exception as e:
            error_msg = f"Analysis cycle failed: {str(e)}"
            logger.error(error_msg)
            cycle.errors.append(error_msg)
            cycle.success = False
            self._handle_error(error_msg)
        
        finally:
            cycle.end_time = datetime.utcnow()
            cycle.duration_seconds = (cycle.end_time - cycle.start_time).total_seconds()
            self._complete_cycle(cycle)
            
            logger.info(
                f"Analysis cycle #{cycle_id} completed in {cycle.duration_seconds:.1f}s "
                f"({cycle.tickers_processed} tickers, {cycle.alerts_generated} alerts, "
                f"{cycle.trades_executed} trades)"
            )

    async def _force_trade_for_cycle(self):
        """Place at least one small trade when aggressive mode is enabled and no trades were executed."""
        # Build ranked candidate list: (ticker, action, score, alert_obj_or_None)
        alerts = self.alert_system.get_active_alerts()
        candidates: List[tuple] = []

        def is_at_max_position(ticker: str) -> bool:
            if ticker in self.trading_engine.positions:
                pos = self.trading_engine.positions[ticker]
                max_val = self.trading_engine.portfolio_value * self.trading_engine.max_single_position
                return abs(pos.market_value) >= max_val
            return False

        # 1) Alerts by confidence
        if config.aggressive_prefer_alerts and alerts:
            normalized_alerts: List[TradingAlert] = []
            for a in alerts:
                if a.processed:
                    continue
                if a.recommended_action == "monitor":
                    try:
                        sig = a.source_data.get("volatility_signal", {}) if a.source_data else {}
                        direction = sig.get("direction")
                        if direction == "bullish":
                            a.recommended_action = "buy"
                        elif direction == "bearish":
                            a.recommended_action = "sell"
                    except Exception:
                        pass
                if a.recommended_action in ("buy", "sell") and len(a.ticker) >= 3:
                    score = a.confidence + 0.1 * a.volatility_score + 0.01 * min(a.article_count, 50)
                    candidates.append((a.ticker, a.recommended_action, score, a))

        # 2) Data-driven ranking (relevance + sentiment confidence)
        for t in config.stock_tickers:
            if len(t) < 3:
                continue
            history = self.data_processor.historical_data.get(t, [])
            if not history:
                continue
            last = history[-1]
            conf = (last.sentiment_metrics.confidence_score if last.sentiment_metrics else 0.0)
            relevance = getattr(last, "relevance_score", 0.0)
            score = 0.6 * relevance + 0.4 * conf
            if last.sentiment_metrics:
                if last.sentiment_metrics.sentiment_change != 0:
                    action = "buy" if last.sentiment_metrics.sentiment_change > 0 else "sell"
                else:
                    action = "buy" if last.sentiment_metrics.average_sentiment >= 0 else "sell"
            else:
                action = "buy"
            candidates.append((t, action, score, None))

        # Deduplicate by ticker keeping highest score
        best_by_ticker: Dict[str, tuple] = {}
        for t, action, score, aobj in candidates:
            if t not in best_by_ticker or score > best_by_ticker[t][2]:
                best_by_ticker[t] = (t, action, score, aobj)
        ranked = list(best_by_ticker.values())
        # Filter out tickers already at max position to diversify
        ranked = [c for c in ranked if not is_at_max_position(c[0])]
        # Sort by score desc; break ties by least recently traded
        ranked.sort(key=lambda x: (x[2], -(self.trading_engine.last_trade_time.get(x[0]).timestamp() if self.trading_engine.last_trade_time.get(x[0]) else 0)), reverse=True)

        selected_ticker: Optional[str] = None
        fallback_action: str = "buy"
        alert_obj = None

        # Try candidates with full engine evaluation; pick the first that passes
        for t, act, _score, aobj in ranked:
            try:
                price_try = await self.alpaca_client.get_current_price(t)
                if not price_try:
                    continue
                use_alert = aobj
                if not use_alert:
                    # create synthetic alert for evaluation
                    from .alert_system import AlertLevel, AlertType, TradingAlert
                    import hashlib
                    aid = hashlib.md5(f"{t}_{int(datetime.utcnow().timestamp())}".encode()).hexdigest()[:12]
                    use_alert = TradingAlert(
                        alert_id=aid,
                        ticker=t,
                        alert_type=AlertType.TRADING_OPPORTUNITY,
                        alert_level=AlertLevel.MEDIUM,
                        title=f"Aggressive Candidate - {t}",
                        description="Synthetic alert for aggressive selection",
                        timestamp=datetime.utcnow(),
                        expires_at=datetime.utcnow() + timedelta(minutes=10),
                        sentiment_score=0.0,
                        sentiment_change=0.0,
                        volatility_score=0.5,
                        confidence=max(0.5, float(config.aggressive_confidence)),
                        article_count=0,
                        top_headlines=[],
                        key_themes=[],
                        recommended_action=act,
                        position_size_recommendation=float(config.aggressive_min_trade_value),
                        stop_loss_suggestion=config.stop_loss_percentage,
                        take_profit_suggestion=config.take_profit_percentage,
                        source_data={}
                    )
                decision_try = await self.trading_engine.evaluate_trading_decision(use_alert, price_try)
                if decision_try:
                    selected_ticker, fallback_action, alert_obj = t, act, use_alert
                    current_price = price_try
                    decision = decision_try
                    # Execute immediately
                    order_id = await self.alpaca_client.execute_trading_decision(decision)
                    if order_id:
                        logger.info(f"Aggressive mode trade executed for {selected_ticker}: {order_id}")
                        self.trading_engine.update_position(
                            decision.ticker,
                            decision.quantity,
                            current_price,
                            decision.action,
                            stop_loss=decision.stop_loss,
                            take_profit=decision.take_profit
                        )
                        if self.current_cycle:
                            self.current_cycle.trades_executed += 1
                        return
            except Exception:
                continue

        # If we get here, no candidate passed engine checks. Force minimal trade on the top-ranked viable ticker.
        if ranked:
            selected_ticker, fallback_action, _score, alert_obj = ranked[0]
        else:
            # Final fallback: first eligible ticker
            for t in config.stock_tickers:
                if len(t) >= 3 and not is_at_max_position(t):
                    selected_ticker = t
                    fallback_action = "buy"
                    break
        
        if not selected_ticker:
            logger.warning("Aggressive mode: no suitable ticker found for forced trade")
            return
        
        # Fetch current price
        current_price = await self.alpaca_client.get_current_price(selected_ticker)
        if not current_price:
            logger.warning(f"Aggressive mode: unable to fetch price for {selected_ticker}")
            return
        
        # Compute minimal quantity based on configured min trade value
        min_value = max(1.0, float(config.aggressive_min_trade_value))
        quantity = max(1, int(min_value // max(0.01, current_price)))
        
        # If quantity is still 0 due to high price, buy 1 share
        if quantity <= 0:
            quantity = 1
        
        # Build a lightweight synthetic alert if none chosen
        from .alert_system import AlertLevel, AlertType, TradingAlert
        from .trading_engine import TradeAction, OrderType, TradingDecision
        from datetime import datetime, timedelta
        import hashlib
        
        if not alert_obj:
            aid = hashlib.md5(f"{selected_ticker}_{int(datetime.utcnow().timestamp())}".encode()).hexdigest()[:12]
            alert_obj = TradingAlert(
                alert_id=aid,
                ticker=selected_ticker,
                alert_type=AlertType.TRADING_OPPORTUNITY,
                alert_level=AlertLevel.MEDIUM,
                title=f"Aggressive Mode Trade - {selected_ticker}",
                description="Forcing a minimal trade to satisfy aggressive mode.",
                timestamp=datetime.utcnow(),
                expires_at=datetime.utcnow() + timedelta(minutes=10),
                sentiment_score=0.0,
                sentiment_change=0.0,
                volatility_score=0.6,
                confidence=max(0.5, config.aggressive_confidence),
                article_count=0,
                top_headlines=[],
                key_themes=[],
                recommended_action=fallback_action,
                position_size_recommendation=float(config.aggressive_min_trade_value),
                stop_loss_suggestion=config.stop_loss_percentage,
                take_profit_suggestion=config.take_profit_percentage,
                source_data={}
            )
        
        # Evaluate trading decision using engine for consistency/risk levels
        decision = await self.trading_engine.evaluate_trading_decision(alert_obj, current_price)
        
        if not decision:
            # Construct a minimal market order decision bypassing some conservatism
            action_enum = TradeAction.BUY if fallback_action == "buy" else TradeAction.SELL
            stop_loss = current_price * (1 - config.stop_loss_percentage) if action_enum == TradeAction.BUY else current_price * (1 + config.stop_loss_percentage)
            take_profit = current_price * (1 + config.take_profit_percentage) if action_enum == TradeAction.BUY else current_price * (1 - config.take_profit_percentage)
            decision = TradingDecision(
                ticker=selected_ticker,
                action=action_enum,
                quantity=quantity,
                order_type=OrderType.MARKET,
                price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                confidence=max(0.5, alert_obj.confidence),
                reasoning="Aggressive mode forced trade to meet minimum trades per cycle",
                risk_score=0.5,
                expected_return=0.0,
                max_risk=quantity * current_price * config.stop_loss_percentage,
                alert_id=alert_obj.alert_id,
                signal_strength=alert_obj.volatility_score,
                sentiment_score=alert_obj.sentiment_score,
                decision_time=datetime.utcnow(),
                valid_until=datetime.utcnow() + timedelta(minutes=10)
            )
        
        # Execute the trade
        order_id = await self.alpaca_client.execute_trading_decision(decision)
        if order_id:
            logger.info(f"Aggressive mode trade executed for {selected_ticker}: {order_id}")
            self.trading_engine.update_position(
                decision.ticker,
                decision.quantity,
                current_price,
                decision.action,
                stop_loss=decision.stop_loss,
                take_profit=decision.take_profit
            )
            if self.current_cycle:
                self.current_cycle.trades_executed += 1
    
    async def _process_ticker(self, ticker: str, cycle: CycleMetrics):
        """Process a single ticker through the complete pipeline."""
        
        # Get company name for the ticker (comprehensive mapping)
        company_names = {
            # Tech Giants
            "AAPL": "Apple Inc",
            "MSFT": "Microsoft Corporation", 
            "GOOGL": "Alphabet Inc",
            "AMZN": "Amazon.com Inc",
            "TSLA": "Tesla Inc",
            "META": "Meta Platforms Inc",
            "NVDA": "NVIDIA Corporation",
            "NFLX": "Netflix Inc",
            "BABA": "Alibaba Group",
            "V": "Visa Inc",
            # Additional Tech
            "ADBE": "Adobe Inc",
            "CRM": "Salesforce Inc",
            "ORCL": "Oracle Corporation",
            "INTC": "Intel Corporation",
            "AMD": "Advanced Micro Devices",
            "QCOM": "Qualcomm Inc",
            "AVGO": "Broadcom Inc",
            "TXN": "Texas Instruments",
            "MU": "Micron Technology",
            "KLAC": "KLA Corporation",
            # Financial Services
            "JPM": "JPMorgan Chase",
            "BAC": "Bank of America",
            "WFC": "Wells Fargo",
            "GS": "Goldman Sachs",
            "MS": "Morgan Stanley",
            "C": "Citigroup",
            "USB": "US Bancorp",
            "PNC": "PNC Financial",
            "TFC": "Truist Financial",
            "COF": "Capital One",
            # Healthcare
            "JNJ": "Johnson & Johnson",
            "PFE": "Pfizer Inc",
            "UNH": "UnitedHealth Group",
            "ABBV": "AbbVie Inc",
            "TMO": "Thermo Fisher Scientific",
            "DHR": "Danaher Corporation",
            "LLY": "Eli Lilly",
            "MRK": "Merck & Co",
            "BMY": "Bristol Myers Squibb",
            "AMGN": "Amgen Inc",
            # Consumer
            "PG": "Procter & Gamble",
            "KO": "Coca-Cola Company",
            "PEP": "PepsiCo Inc",
            "WMT": "Walmart Inc",
            "HD": "Home Depot",
            "MCD": "McDonald's Corporation",
            "SBUX": "Starbucks Corporation",
            "NKE": "Nike Inc",
            "DIS": "Walt Disney Company",
            "CMCSA": "Comcast Corporation",
            # Energy
            "XOM": "Exxon Mobil",
            "CVX": "Chevron Corporation",
            "COP": "ConocoPhillips",
            "EOG": "EOG Resources",
            "SLB": "Schlumberger",
            "PSX": "Phillips 66",
            "VLO": "Valero Energy",
            "MPC": "Marathon Petroleum",
            "OXY": "Occidental Petroleum",
            "KMI": "Kinder Morgan",
            # Industrial
            "CAT": "Caterpillar Inc",
            "BA": "Boeing Company",
            "MMM": "3M Company",
            "GE": "General Electric",
            "HON": "Honeywell International",
            "UPS": "United Parcel Service",
            "FDX": "FedEx Corporation",
            "LMT": "Lockheed Martin",
            "RTX": "Raytheon Technologies",
            "DE": "Deere & Company",
            # Communication
            "T": "AT&T Inc",
            "VZ": "Verizon Communications",
            "TMUS": "T-Mobile US",
            "CHTR": "Charter Communications",
            "CME": "CME Group",
            "ICE": "Intercontinental Exchange",
            "SPGI": "S&P Global",
            "MCO": "Moody's Corporation",
            "BLK": "BlackRock Inc",
            "SCHW": "Charles Schwab",
            # Real Estate
            "PLD": "Prologis Inc",
            "AMT": "American Tower",
            "CCI": "Crown Castle",
            "EQIX": "Equinix Inc",
            "DLR": "Digital Realty Trust",
            "PSA": "Public Storage",
            "SPG": "Simon Property Group",
            "O": "Realty Income",
            "AVB": "AvalonBay Communities",
            "EQR": "Equity Residential",
            # Materials
            "LIN": "Linde PLC",
            "APD": "Air Products",
            "FCX": "Freeport-McMoRan",
            "NEM": "Newmont Corporation",
            "DOW": "Dow Inc",
            "DD": "DuPont de Nemours",
            "NUE": "Nucor Corporation",
            "X": "United States Steel",
            "BLL": "Ball Corporation",
            "ALB": "Albemarle Corporation"
        }
        
        company_name = company_names.get(ticker, ticker)
        
        # Fetch GDELT data
        gdelt_data = await self.gdelt_client.get_stock_related_news(
            ticker=ticker,
            company_name=company_name,
            hours_back=config.rolling_window_minutes // 60
        )
        
        # Process the data
        event_data = self.data_processor.process_gdelt_response(gdelt_data)
        
        # Analyze volatility
        volatility_signal = self.volatility_analyzer.analyze_volatility(event_data)
        
        # Generate alerts
        alerts = self.alert_system.process_event_data(event_data, volatility_signal)
        cycle.alerts_generated += len(alerts)
        
        # Log ticker processing results
        logger.debug(
            f"Processed {ticker}: {event_data.event_volume} articles, "
            f"volatility: {volatility_signal.strength:.3f}, "
            f"alerts: {len(alerts)}"
        )
    
    async def _process_pending_alerts(self):
        """Process pending alerts and execute trades if appropriate."""
        if self.trading_engine.should_halt_trading():
            logger.warning("Trading halted due to risk conditions")
            return
        
        active_alerts = self.alert_system.get_active_alerts()
        
        for alert in active_alerts:
            if alert.processed:
                continue
            
            # Convert non-actionable 'monitor' into a directional action when possible
            if alert.recommended_action == "monitor":
                try:
                    sig = alert.source_data.get("volatility_signal", {}) if alert.source_data else {}
                    direction = sig.get("direction")
                    if direction == "bullish":
                        alert.recommended_action = "buy"
                    elif direction == "bearish":
                        alert.recommended_action = "sell"
                except Exception:
                    pass
            
            if alert.recommended_action == "hold":
                continue
            
            try:
                # Get current price
                current_price = await self.alpaca_client.get_current_price(alert.ticker)
                if not current_price:
                    logger.warning(f"Cannot get current price for {alert.ticker}")
                    continue
                
                # Evaluate trading decision
                decision = await self.trading_engine.evaluate_trading_decision(
                    alert, current_price
                )
                
                if decision:
                    # Execute the trade
                    order_id = await self.alpaca_client.execute_trading_decision(decision)
                    
                    if order_id:
                        logger.info(f"Trade executed for {alert.ticker}: {order_id}")
                        
                        # Update trading engine position tracking
                        self.trading_engine.update_position(
                            decision.ticker,
                            decision.quantity,
                            current_price,
                            decision.action,
                            stop_loss=decision.stop_loss,
                            take_profit=decision.take_profit
                        )
                        
                        if self.current_cycle:
                            self.current_cycle.trades_executed += 1
                    
                    # Mark alert as processed
                    alert.processed = True
                
            except Exception as e:
                logger.error(f"Error processing alert {alert.alert_id}: {e}")
                continue
    
    async def _update_portfolio_positions(self):
        """Update portfolio positions from Alpaca, merging to preserve SL/TP."""
        try:
            positions = await self.alpaca_client.get_positions()
            
            # Update trading engine with current positions (merge)
            from .trading_engine import Position as EnginePosition
            updated_tickers: set = set()
            for p in positions:
                updated_tickers.add(p.ticker)
                if p.ticker in self.trading_engine.positions:
                    ep = self.trading_engine.positions[p.ticker]
                    ep.quantity = p.quantity
                    ep.current_price = p.current_price
                    if not ep.entry_price:
                        ep.entry_price = p.cost_basis / max(p.quantity, 1) if p.quantity else 0.0
                else:
                    self.trading_engine.positions[p.ticker] = EnginePosition(
                        ticker=p.ticker,
                        quantity=p.quantity,
                        entry_price=p.cost_basis / max(p.quantity, 1) if p.quantity else 0.0,
                        current_price=p.current_price,
                        entry_time=datetime.utcnow(),
                        stop_loss=None,
                        take_profit=None
                    )
            # Remove positions not present in Alpaca
            for ticker in list(self.trading_engine.positions.keys()):
                if ticker not in updated_tickers:
                    del self.trading_engine.positions[ticker]
            
            # Update status
            self.status.open_positions = len(positions)
            
        except Exception as e:
            logger.error(f"Error updating portfolio positions: {e}")

    async def _monitor_portfolio(self):
        """Monitor portfolio and update orders."""
        try:
            # Update pending orders
            await self.alpaca_client.update_pending_orders()
            
            # Get portfolio performance
            performance = await self.alpaca_client.get_portfolio_performance()
            
            # Update status
            if performance:
                self.status.daily_pnl = performance.get("total_unrealized_pnl", 0)
                self.status.performance_metrics = performance
            
            # Enforce exits when SL/TP hit (refresh latest prices first)
            await self._refresh_position_prices()
            await self._enforce_position_exits()
            # Also enforce exits based on broker PnL percentages
            await self._enforce_exits_by_pnl()
            
            # Check for risk management triggers
            if self.trading_engine.should_halt_trading():
                logger.warning("Risk management triggered - halting trading")
            
        except Exception as e:
            logger.error(f"Error monitoring portfolio: {e}")

    async def _enforce_position_exits(self):
        """Close positions when take-profit or stop-loss thresholds are reached."""
        try:
            tp_pct = float(config.take_profit_percentage)
            sl_pct = float(config.stop_loss_percentage)
            for ticker, pos in list(self.trading_engine.positions.items()):
                if pos.quantity == 0 or pos.entry_price == 0:
                    continue
                current_price = pos.current_price
                entry_price = pos.entry_price
                is_long = pos.quantity > 0
                # Compute thresholds
                use_config = bool(config.enforce_config_exits)
                sl_price = (
                    entry_price * (1 - sl_pct) if is_long else entry_price * (1 + sl_pct)
                ) if (use_config or not pos.stop_loss) else pos.stop_loss
                tp_price = (
                    entry_price * (1 + tp_pct) if is_long else entry_price * (1 - tp_pct)
                ) if (use_config or not pos.take_profit) else pos.take_profit
                hit_tp = current_price >= tp_price if is_long else current_price <= tp_price
                hit_sl = current_price <= sl_price if is_long else current_price >= sl_price
                if hit_tp or hit_sl:
                    reason = "take-profit" if hit_tp else "stop-loss"
                    logger.info(
                        f"{reason.title()} reached for {ticker} at {current_price:.4f} (entry {entry_price:.4f}, "
                        f"tp={tp_price:.4f}, sl={sl_price:.4f}, long={is_long}). Closing position."
                    )
                    try:
                        await self.alpaca_client._cancel_open_orders_for_symbol(ticker)
                    except Exception:
                        pass
                    await self.alpaca_client.close_position(ticker, percentage=1.0)
                    # Remove from engine after close; PnL captured by broker
                    if ticker in self.trading_engine.positions:
                        del self.trading_engine.positions[ticker]
        except Exception as e:
            logger.error(f"Error enforcing position exits: {e}")

    async def _enforce_exits_by_pnl(self):
        """Close positions using Alpaca's unrealized P&L percentage (plpc)."""
        try:
            tp_pct = float(config.take_profit_percentage)
            sl_pct = float(config.stop_loss_percentage)
            logger.info(f"Checking P&L exits: TP={tp_pct:.4f}, SL={sl_pct:.4f}")
            
            positions = await self.alpaca_client.get_positions()
            logger.info(f"Found {len(positions)} positions to check for P&L exits")
            
            for p in positions:
                # unrealized_pnl_percent stored as percent (e.g., 1.0 for 1%)
                plpc = (p.unrealized_pnl_percent / 100.0) if p.unrealized_pnl_percent is not None else 0.0
                qty = int(p.quantity)
                if qty == 0:
                    continue
                
                # For long positions: positive plpc is profit, negative is loss
                # For short positions: negative plpc is profit, positive is loss
                is_long = qty > 0
                
                # Check if we should exit based on P&L
                should_exit = False
                reason = ""
                
                if is_long:
                    # Long position: exit if profit >= tp_pct OR loss <= -sl_pct
                    if plpc >= tp_pct:
                        should_exit = True
                        reason = "TP"
                    elif plpc <= -sl_pct:
                        should_exit = True
                        reason = "SL"
                else:
                    # Short position: exit if profit >= tp_pct OR loss <= -sl_pct
                    if plpc <= -tp_pct:
                        should_exit = True
                        reason = "TP"
                    elif plpc >= sl_pct:
                        should_exit = True
                        reason = "SL"
                
                if should_exit:
                    logger.info(
                        f"PnL exit {reason} for {p.ticker}: plpc={plpc:.4f}, tp={tp_pct:.4f}, sl={sl_pct:.4f}, long={is_long}. Closing position."
                    )
                    try:
                        await self.alpaca_client._cancel_open_orders_for_symbol(p.ticker)
                    except Exception:
                        pass
                    await self.alpaca_client.close_position(p.ticker, percentage=1.0)
                    if p.ticker in self.trading_engine.positions:
                        del self.trading_engine.positions[p.ticker]
                else:
                    # Debug logging to see what's happening
                    logger.debug(
                        f"PnL check for {p.ticker}: plpc={plpc:.4f}, tp={tp_pct:.4f}, sl={sl_pct:.4f}, long={is_long}, should_exit={should_exit}"
                    )
        except Exception as e:
            logger.error(f"Error enforcing P&L exits: {e}")

    async def _refresh_position_prices(self):
        """Refresh engine position prices from Alpaca quotes."""
        try:
            for ticker, pos in self.trading_engine.positions.items():
                latest = await self.alpaca_client.get_current_price(ticker)
                if latest:
                    pos.current_price = latest
        except Exception as e:
            logger.warning(f"Price refresh failed: {e}")
    
    def _schedule_portfolio_monitoring(self):
        """Schedule portfolio monitoring."""
        if not self.is_running:
            return
        
        asyncio.create_task(self._monitor_portfolio())
    
    def _schedule_health_check(self):
        """Schedule system health check."""
        if not self.is_running:
            return
        
        asyncio.create_task(self._perform_health_check())
    
    async def _perform_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        health_results = {}
        overall_health = "healthy"
        
        for check_name, check_func in self.health_checks.items():
            try:
                result = await check_func()
                health_results[check_name] = result
                
                if result.get("status") == "error":
                    overall_health = "error"
                elif result.get("status") == "warning" and overall_health != "error":
                    overall_health = "warning"
                    
            except Exception as e:
                health_results[check_name] = {
                    "status": "error",
                    "message": f"Health check failed: {str(e)}"
                }
                overall_health = "error"
        
        health_results["overall_health"] = overall_health
        health_results["timestamp"] = datetime.utcnow().isoformat()
        
        # Update system status
        self.status.system_health = overall_health
        
        if overall_health == "error":
            logger.error("System health check failed")
        elif overall_health == "warning":
            logger.warning("System health check shows warnings")
        
        return health_results
    
    async def _check_gdelt_api_health(self) -> Dict[str, Any]:
        """Check GDELT API health."""
        try:
            # Use adaptive news fetch logic for a representative ticker
            test_data = await self.gdelt_client.get_stock_related_news(
                ticker="AAPL",
                company_name="Apple Inc",
                hours_back=2
            )

            num_articles = len(test_data.get("articles", [])) if isinstance(test_data, dict) else 0
            if num_articles > 0:
                return {"status": "healthy", "message": f"GDELT API responding ({num_articles} articles)"}
            else:
                return {"status": "error", "message": "GDELT API returned 0 articles for test query"}
                
        except Exception as e:
            return {"status": "error", "message": f"GDELT API error: {str(e)}"}
    
    async def _check_alpaca_api_health(self) -> Dict[str, Any]:
        """Check Alpaca API health."""
        try:
            account_info = await self.alpaca_client.get_account_info()
            
            if account_info and account_info.get("status") == "ACTIVE":
                return {"status": "healthy", "message": "Alpaca API responding"}
            else:
                logger.warning(f"Alpaca account status: {account_info.get('status') if account_info else 'No account info'}")
                return {"status": "error", "message": f"Alpaca account status: {account_info.get('status') if account_info else 'No account info'}"}
                
        except Exception as e:
            logger.error(f"Alpaca API health check error: {e}")
            return {"status": "error", "message": f"Alpaca API error: {str(e)}"}
    
    async def _check_data_processor_health(self) -> Dict[str, Any]:
        """Check data processor health."""
        try:
            # Check if data processor has recent data
            total_tickers = len(config.stock_tickers)
            tickers_with_data = len([
                ticker for ticker in config.stock_tickers
                if ticker in self.data_processor.historical_data
            ])
            
            if tickers_with_data >= total_tickers * 0.8:  # 80% coverage
                return {"status": "healthy", "message": f"Data coverage: {tickers_with_data}/{total_tickers}"}
            elif tickers_with_data >= total_tickers * 0.5:  # 50% coverage
                return {"status": "warning", "message": f"Low data coverage: {tickers_with_data}/{total_tickers}"}
            else:
                return {"status": "error", "message": f"Very low data coverage: {tickers_with_data}/{total_tickers}"}
                
        except Exception as e:
            return {"status": "error", "message": f"Data processor error: {str(e)}"}
    
    async def _check_trading_engine_health(self) -> Dict[str, Any]:
        """Check trading engine health."""
        try:
            if self.trading_engine.should_halt_trading():
                return {"status": "error", "message": "Trading halted due to risk conditions"}
            
            portfolio_summary = self.trading_engine.get_portfolio_summary()
            
            if portfolio_summary.get("consecutive_losses", 0) >= 3:
                return {"status": "warning", "message": "High consecutive losses"}
            
            return {"status": "healthy", "message": "Trading engine operating normally"}
            
        except Exception as e:
            return {"status": "error", "message": f"Trading engine error: {str(e)}"}
    
    async def _check_disk_space(self) -> Dict[str, Any]:
        """Check available disk space."""
        try:
            import shutil
            total, used, free = shutil.disk_usage("/")
            
            free_percent = free / total * 100
            
            if free_percent < 5:
                return {"status": "error", "message": f"Low disk space: {free_percent:.1f}% free"}
            elif free_percent < 15:
                return {"status": "warning", "message": f"Disk space getting low: {free_percent:.1f}% free"}
            else:
                return {"status": "healthy", "message": f"Disk space: {free_percent:.1f}% free"}
                
        except Exception as e:
            return {"status": "error", "message": f"Disk space check error: {str(e)}"}
    
    async def _check_memory_usage(self) -> Dict[str, Any]:
        """Check memory usage."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            
            if memory.percent > 90:
                return {"status": "error", "message": f"High memory usage: {memory.percent:.1f}%"}
            elif memory.percent > 80:
                return {"status": "warning", "message": f"Memory usage elevated: {memory.percent:.1f}%"}
            else:
                return {"status": "healthy", "message": f"Memory usage: {memory.percent:.1f}%"}
                
        except ImportError:
            return {"status": "warning", "message": "psutil not available for memory monitoring"}
        except Exception as e:
            return {"status": "error", "message": f"Memory check error: {str(e)}"}
    
    def _schedule_daily_reset(self):
        """Schedule daily reset and cleanup."""
        if not self.is_running:
            return
        
        asyncio.create_task(self._daily_reset())
    
    async def _daily_reset(self):
        """Perform daily reset and cleanup."""
        try:
            logger.info("Performing daily reset")
            
            # Reset trading engine daily metrics
            self.trading_engine.reset_daily_metrics()
            
            # Reset consecutive errors
            self.consecutive_errors = 0
            
            # Clean up old cycle history
            if len(self.cycle_history) > 100:
                self.cycle_history = self.cycle_history[-50:]
            
            # Clean up old performance history
            if len(self.performance_history) > 24 * 7:  # Keep 7 days of hourly data
                self.performance_history = self.performance_history[-24*7:]
            
            logger.info("Daily reset completed")
            
        except Exception as e:
            logger.error(f"Error during daily reset: {e}")
    
    def _schedule_market_hours_check(self):
        """Schedule market hours check."""
        if not self.is_running:
            return
        
        asyncio.create_task(self._check_market_hours())
    
    async def _check_market_hours(self):
        """Check market hours and adjust system behavior."""
        try:
            market_hours = await self.alpaca_client.get_market_hours()
            
            if not market_hours.get("is_open", False):
                logger.info("Market is closed - reducing system activity")
                # Could implement reduced activity mode here
            
        except Exception as e:
            logger.error(f"Error checking market hours: {e}")
    
    def _schedule_performance_logging(self):
        """Schedule performance logging."""
        if not self.is_running:
            return
        
        asyncio.create_task(self._log_performance_metrics())
    
    async def _log_performance_metrics(self):
        """Log performance metrics."""
        try:
            # Collect metrics
            portfolio_summary = self.trading_engine.get_portfolio_summary()
            alert_summary = self.alert_system.get_alert_summary()
            
            performance_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "system_status": {
                    "is_running": self.status.is_running,
                    "total_cycles": self.status.total_cycles,
                    "successful_cycles": self.status.successful_cycles,
                    "system_health": self.status.system_health
                },
                "portfolio": portfolio_summary,
                "alerts": alert_summary,
                "recent_cycle_performance": {
                    "avg_duration": statistics.mean([
                        c.duration_seconds for c in self.cycle_history[-10:]
                        if c.duration_seconds is not None
                    ]) if self.cycle_history else 0,
                    "success_rate": sum(1 for c in self.cycle_history[-10:] if c.success) / min(len(self.cycle_history), 10) * 100
                }
            }
            
            self.performance_history.append(performance_data)
            
            logger.info(f"Performance logged: Portfolio value: ${portfolio_summary.get('portfolio_value', 0):.2f}, "
                       f"Active alerts: {alert_summary.get('active_alerts_count', 0)}, "
                       f"System health: {self.status.system_health}")
            
        except Exception as e:
            logger.error(f"Error logging performance metrics: {e}")
    
    def _complete_cycle(self, cycle: CycleMetrics):
        """Complete a cycle and update statistics."""
        self.cycle_history.append(cycle)
        self.status.total_cycles += 1
        
        if cycle.success:
            self.status.successful_cycles += 1
            self.consecutive_errors = 0
        else:
            self.status.failed_cycles += 1
        
        self.status.last_update = cycle.end_time
        self.status.active_alerts = len(self.alert_system.get_active_alerts())
        
        self.current_cycle = None
    
    def _handle_error(self, error_message: str):
        """Handle system errors."""
        self.consecutive_errors += 1
        self.last_error_time = datetime.utcnow()
        
        # Add to status errors (keep last 10)
        self.status.errors.append(f"{datetime.utcnow().isoformat()}: {error_message}")
        if len(self.status.errors) > 10:
            self.status.errors.pop(0)
        
        # Update system health
        if self.consecutive_errors >= 3:
            self.status.system_health = "error"
        elif self.consecutive_errors >= 1:
            self.status.system_health = "warning"
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            "status": {
                "is_running": self.status.is_running,
                "system_health": self.status.system_health,
                "last_update": self.status.last_update.isoformat() if self.status.last_update else None,
                "consecutive_errors": self.consecutive_errors
            },
            "statistics": {
                "total_cycles": self.status.total_cycles,
                "successful_cycles": self.status.successful_cycles,
                "failed_cycles": self.status.failed_cycles,
                "success_rate": (self.status.successful_cycles / max(self.status.total_cycles, 1)) * 100
            },
            "current_cycle": {
                "cycle_id": self.current_cycle.cycle_id if self.current_cycle else None,
                "start_time": self.current_cycle.start_time.isoformat() if self.current_cycle else None,
                "tickers_processed": self.current_cycle.tickers_processed if self.current_cycle else 0
            } if self.current_cycle else None,
            "alerts": {
                "active_count": self.status.active_alerts,
                "summary": self.alert_system.get_alert_summary()
            },
            "portfolio": {
                "open_positions": self.status.open_positions,
                "daily_pnl": self.status.daily_pnl,
                "summary": self.trading_engine.get_portfolio_summary()
            },
            "errors": self.status.errors[-5:],  # Last 5 errors
            "performance_metrics": self.status.performance_metrics
        }
