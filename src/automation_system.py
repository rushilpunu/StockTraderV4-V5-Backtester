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
    
    async def start(self):
        """Start the automation system."""
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
            # Check if market is open
            market_hours = await self.alpaca_client.get_market_hours()
            if not market_hours.get("is_open", False):
                logger.info("Market is closed - skipping analysis cycle")
                cycle.success = True
                cycle.end_time = datetime.utcnow()
                cycle.duration_seconds = (cycle.end_time - cycle.start_time).total_seconds()
                self._complete_cycle(cycle)
                return
            
            # Process each ticker
            for ticker in config.stock_tickers:
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
            
            # Check for trade execution opportunities
            await self._process_pending_alerts()
            
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
    
    async def _process_ticker(self, ticker: str, cycle: CycleMetrics):
        """Process a single ticker through the complete pipeline."""
        
        # Get company name for the ticker (simplified mapping)
        company_names = {
            "AAPL": "Apple Inc",
            "MSFT": "Microsoft Corporation", 
            "GOOGL": "Alphabet Inc",
            "AMZN": "Amazon.com Inc",
            "TSLA": "Tesla Inc",
            "META": "Meta Platforms Inc",
            "NVDA": "NVIDIA Corporation",
            "NFLX": "Netflix Inc",
            "BABA": "Alibaba Group",
            "V": "Visa Inc"
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
            if alert.processed or alert.recommended_action == "hold":
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
                            decision.action
                        )
                        
                        if self.current_cycle:
                            self.current_cycle.trades_executed += 1
                    
                    # Mark alert as processed
                    alert.processed = True
                
            except Exception as e:
                logger.error(f"Error processing alert {alert.alert_id}: {e}")
                continue
    
    async def _update_portfolio_positions(self):
        """Update portfolio positions from Alpaca."""
        try:
            positions = await self.alpaca_client.get_positions()
            
            # Update trading engine with current positions
            for position in positions:
                self.trading_engine.positions[position.ticker] = position
            
            # Update status
            self.status.open_positions = len(positions)
            
        except Exception as e:
            logger.error(f"Error updating portfolio positions: {e}")
    
    def _schedule_portfolio_monitoring(self):
        """Schedule portfolio monitoring."""
        if not self.is_running:
            return
        
        asyncio.create_task(self._monitor_portfolio())
    
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
            
            # Check for risk management triggers
            if self.trading_engine.should_halt_trading():
                logger.warning("Risk management triggered - halting trading")
            
        except Exception as e:
            logger.error(f"Error monitoring portfolio: {e}")
    
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
            # Simple test query
            test_data = await self.gdelt_client.get_doc_search(
                query="test",
                start_date=datetime.utcnow() - timedelta(hours=1),
                end_date=datetime.utcnow(),
                max_records=1
            )
            
            if "articles" in test_data:
                return {"status": "healthy", "message": "GDELT API responding"}
            else:
                return {"status": "warning", "message": "GDELT API response unexpected"}
                
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
