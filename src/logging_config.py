"""Comprehensive logging configuration and error handling."""

import os
import sys
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
import json
import traceback
from loguru import logger

from .config import config


class LoggingSetup:
    """Setup comprehensive logging for the trading system."""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.log_dir / "main").mkdir(exist_ok=True)
        (self.log_dir / "trades").mkdir(exist_ok=True)
        (self.log_dir / "alerts").mkdir(exist_ok=True)
        (self.log_dir / "errors").mkdir(exist_ok=True)
        (self.log_dir / "performance").mkdir(exist_ok=True)
        
        self._setup_loggers()
    
    def _setup_loggers(self):
        """Configure all loggers."""
        # Remove default logger
        logger.remove()
        
        # Console logger
        logger.add(
            sys.stdout,
            level=config.log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                   "<level>{level: <8}</level> | "
                   "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                   "<level>{message}</level>",
            colorize=True,
            backtrace=True,
            diagnose=True
        )
        
        # Main application log
        logger.add(
            self.log_dir / "main" / "app_{time:YYYY-MM-DD}.log",
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
            rotation="1 day",
            retention="30 days",
            compression="zip",
            backtrace=True,
            diagnose=True,
            enqueue=True
        )
        
        # Error log
        logger.add(
            self.log_dir / "errors" / "errors_{time:YYYY-MM-DD}.log",
            level="ERROR",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
            rotation="1 day",
            retention="90 days",
            compression="zip",
            backtrace=True,
            diagnose=True,
            enqueue=True
        )
        
        # Trading log (for trades and positions)
        self.trade_logger = logger.bind(category="trading")
        logger.add(
            self.log_dir / "trades" / "trading_{time:YYYY-MM-DD}.log",
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}",
            rotation="1 day",
            retention="365 days",  # Keep trading logs for a year
            compression="zip",
            filter=lambda record: record.get("extra", {}).get("category") == "trading",
            enqueue=True
        )
        
        # Alert log
        self.alert_logger = logger.bind(category="alerts")
        logger.add(
            self.log_dir / "alerts" / "alerts_{time:YYYY-MM-DD}.log",
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}",
            rotation="1 day",
            retention="90 days",
            compression="zip",
            filter=lambda record: record.get("extra", {}).get("category") == "alerts",
            enqueue=True
        )
        
        # Performance log
        self.performance_logger = logger.bind(category="performance")
        logger.add(
            self.log_dir / "performance" / "performance_{time:YYYY-MM-DD}.log",
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {message}",
            rotation="1 day",
            retention="90 days",
            compression="zip",
            filter=lambda record: record.get("extra", {}).get("category") == "performance",
            enqueue=True
        )
        
        logger.info("Logging system initialized")
    
    def log_trade_execution(
        self,
        ticker: str,
        action: str,
        quantity: int,
        price: float,
        order_id: str,
        reasoning: str,
        metadata: Dict[str, Any] = None
    ):
        """Log trade execution details."""
        trade_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "ticker": ticker,
            "action": action,
            "quantity": quantity,
            "price": price,
            "order_id": order_id,
            "reasoning": reasoning,
            "metadata": metadata or {}
        }
        
        self.trade_logger.info(f"TRADE_EXECUTION: {json.dumps(trade_data)}")
    
    def log_alert_generated(
        self,
        alert_id: str,
        ticker: str,
        alert_type: str,
        alert_level: str,
        confidence: float,
        metadata: Dict[str, Any] = None
    ):
        """Log alert generation."""
        alert_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "alert_id": alert_id,
            "ticker": ticker,
            "alert_type": alert_type,
            "alert_level": alert_level,
            "confidence": confidence,
            "metadata": metadata or {}
        }
        
        self.alert_logger.info(f"ALERT_GENERATED: {json.dumps(alert_data)}")
    
    def log_performance_metrics(self, metrics: Dict[str, Any]):
        """Log performance metrics."""
        performance_data = {
            "timestamp": datetime.utcnow().isoformat(),
            **metrics
        }
        
        self.performance_logger.info(f"PERFORMANCE_METRICS: {json.dumps(performance_data)}")
    
    def log_api_call(
        self,
        api_name: str,
        endpoint: str,
        method: str,
        status_code: Optional[int] = None,
        response_time_ms: Optional[float] = None,
        error: Optional[str] = None
    ):
        """Log API call details."""
        api_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "api_name": api_name,
            "endpoint": endpoint,
            "method": method,
            "status_code": status_code,
            "response_time_ms": response_time_ms,
            "error": error
        }
        
        if error:
            logger.error(f"API_CALL_ERROR: {json.dumps(api_data)}")
        else:
            logger.debug(f"API_CALL: {json.dumps(api_data)}")
    
    def log_system_event(self, event_type: str, details: Dict[str, Any]):
        """Log system events."""
        event_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            **details
        }
        
        logger.info(f"SYSTEM_EVENT: {json.dumps(event_data)}")


class ErrorHandler:
    """Comprehensive error handling and reporting."""
    
    def __init__(self, logging_setup: LoggingSetup):
        self.logging_setup = logging_setup
        self.error_counts: Dict[str, int] = {}
        self.last_errors: Dict[str, datetime] = {}
        self.max_error_rate = 10  # Max errors per minute per category
    
    def handle_exception(
        self,
        exception: Exception,
        context: str,
        metadata: Dict[str, Any] = None,
        critical: bool = False
    ) -> str:
        """
        Handle exceptions with comprehensive logging and tracking.
        
        Returns:
            Error ID for tracking
        """
        error_id = f"{context}_{int(datetime.utcnow().timestamp())}"
        error_type = type(exception).__name__
        error_message = str(exception)
        
        # Check error rate limiting
        if self._is_error_rate_limited(context):
            logger.warning(f"Error rate limited for context: {context}")
            return error_id
        
        # Log the error
        error_data = {
            "error_id": error_id,
            "error_type": error_type,
            "error_message": error_message,
            "context": context,
            "traceback": traceback.format_exc(),
            "metadata": metadata or {},
            "critical": critical,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if critical:
            logger.critical(f"CRITICAL_ERROR: {json.dumps(error_data)}")
        else:
            logger.error(f"ERROR: {json.dumps(error_data)}")
        
        # Update error tracking
        self._update_error_tracking(context)
        
        # Send alerts for critical errors
        if critical:
            self._send_critical_error_alert(error_data)
        
        return error_id
    
    def _is_error_rate_limited(self, context: str) -> bool:
        """Check if error rate is too high for a context."""
        now = datetime.utcnow()
        
        # Clean old error timestamps
        cutoff = now.timestamp() - 60  # 1 minute ago
        
        error_key = f"{context}_count"
        if error_key in self.error_counts:
            if self.last_errors.get(error_key, now).timestamp() < cutoff:
                self.error_counts[error_key] = 0
        
        # Check current rate
        current_count = self.error_counts.get(error_key, 0)
        return current_count >= self.max_error_rate
    
    def _update_error_tracking(self, context: str):
        """Update error tracking counters."""
        error_key = f"{context}_count"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        self.last_errors[error_key] = datetime.utcnow()
    
    def _send_critical_error_alert(self, error_data: Dict[str, Any]):
        """Send alert for critical errors."""
        # This could be extended to send emails, Slack messages, etc.
        logger.critical(f"CRITICAL ERROR ALERT: {error_data['context']} - {error_data['error_message']}")
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of recent errors."""
        return {
            "error_counts": dict(self.error_counts),
            "last_errors": {
                key: timestamp.isoformat() 
                for key, timestamp in self.last_errors.items()
            },
            "total_errors": sum(self.error_counts.values())
        }


class StructuredLogger:
    """Structured logging for specific components."""
    
    def __init__(self, component_name: str, logging_setup: LoggingSetup):
        self.component_name = component_name
        self.logging_setup = logging_setup
        self.logger = logger.bind(component=component_name)
    
    def log_data_processing(
        self,
        ticker: str,
        articles_processed: int,
        sentiment_score: float,
        volatility_score: float,
        processing_time_ms: float
    ):
        """Log data processing metrics."""
        data = {
            "component": self.component_name,
            "ticker": ticker,
            "articles_processed": articles_processed,
            "sentiment_score": sentiment_score,
            "volatility_score": volatility_score,
            "processing_time_ms": processing_time_ms,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.logger.info(f"DATA_PROCESSING: {json.dumps(data)}")
    
    def log_api_request(
        self,
        api_name: str,
        request_params: Dict[str, Any],
        response_status: str,
        response_time_ms: float,
        data_points: int = 0
    ):
        """Log API request details."""
        data = {
            "component": self.component_name,
            "api_name": api_name,
            "request_params": request_params,
            "response_status": response_status,
            "response_time_ms": response_time_ms,
            "data_points": data_points,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.logger.info(f"API_REQUEST: {json.dumps(data)}")
    
    def log_decision_making(
        self,
        ticker: str,
        decision: str,
        confidence: float,
        reasoning: str,
        input_factors: Dict[str, Any]
    ):
        """Log decision making process."""
        data = {
            "component": self.component_name,
            "ticker": ticker,
            "decision": decision,
            "confidence": confidence,
            "reasoning": reasoning,
            "input_factors": input_factors,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.logger.info(f"DECISION_MAKING: {json.dumps(data)}")
    
    def log_risk_assessment(
        self,
        ticker: str,
        risk_score: float,
        risk_factors: Dict[str, float],
        risk_limits: Dict[str, float],
        action_taken: str
    ):
        """Log risk assessment details."""
        data = {
            "component": self.component_name,
            "ticker": ticker,
            "risk_score": risk_score,
            "risk_factors": risk_factors,
            "risk_limits": risk_limits,
            "action_taken": action_taken,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.logger.info(f"RISK_ASSESSMENT: {json.dumps(data)}")


# Global instances
logging_setup = LoggingSetup()
error_handler = ErrorHandler(logging_setup)


def get_component_logger(component_name: str) -> StructuredLogger:
    """Get a structured logger for a specific component."""
    return StructuredLogger(component_name, logging_setup)


def log_startup_info():
    """Log system startup information."""
    startup_info = {
        "python_version": sys.version,
        "working_directory": os.getcwd(),
        "config": {
            "log_level": config.log_level,
            "trading_enabled": config.trading_enabled,
            "analysis_interval_minutes": config.analysis_interval_minutes,
            "stock_tickers": config.stock_tickers,
            "max_position_size": config.max_position_size
        },
        "environment": {
            "PATH": os.environ.get("PATH", ""),
            "HOME": os.environ.get("HOME", ""),
            "USER": os.environ.get("USER", "unknown")
        }
    }
    
    logging_setup.log_system_event("SYSTEM_STARTUP", startup_info)
    logger.info("System startup logged")


def log_shutdown_info():
    """Log system shutdown information."""
    shutdown_info = {
        "shutdown_time": datetime.utcnow().isoformat(),
        "uptime_seconds": 0,  # Would need to track startup time
        "error_summary": error_handler.get_error_summary()
    }
    
    logging_setup.log_system_event("SYSTEM_SHUTDOWN", shutdown_info)
    logger.info("System shutdown logged")


# Context managers for logging
class LoggedOperation:
    """Context manager for logging operations with timing."""
    
    def __init__(self, operation_name: str, component: str = "system", **metadata):
        self.operation_name = operation_name
        self.component = component
        self.metadata = metadata
        self.start_time = None
        self.logger = get_component_logger(component)
    
    def __enter__(self):
        self.start_time = datetime.utcnow()
        logger.debug(f"Starting operation: {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.utcnow() - self.start_time).total_seconds() * 1000
        
        if exc_type is not None:
            error_id = error_handler.handle_exception(
                exc_val,
                f"{self.component}_{self.operation_name}",
                self.metadata
            )
            logger.error(f"Operation failed: {self.operation_name} (Error ID: {error_id})")
        else:
            logger.debug(f"Operation completed: {self.operation_name} ({duration:.1f}ms)")


# Decorator for automatic error handling
def handle_errors(context: str, critical: bool = False):
    """Decorator for automatic error handling."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_id = error_handler.handle_exception(
                    e, context, {"args": str(args), "kwargs": str(kwargs)}, critical
                )
                logger.error(f"Function {func.__name__} failed with error ID: {error_id}")
                raise
        return wrapper
    return decorator


# Async version of error handling decorator
def handle_async_errors(context: str, critical: bool = False):
    """Decorator for automatic async error handling."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                error_id = error_handler.handle_exception(
                    e, context, {"args": str(args), "kwargs": str(kwargs)}, critical
                )
                logger.error(f"Async function {func.__name__} failed with error ID: {error_id}")
                raise
        return wrapper
    return decorator
