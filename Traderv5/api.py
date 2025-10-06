"""FastAPI application for TraderV5 with backtester integration."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Any

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from Traderv5.configuration import load_trading_parameters, TradingParameters
from Traderv5.backtester import CashAccountBacktester, BacktestResult
from Traderv5.optimized_backtester import OptimizedBacktester
from Traderv5.model.predictor import ModelPredictor

_LOG = logging.getLogger("traderv5.api")

# Global state
_backtester: Optional[CashAccountBacktester] = None
_optimized_backtester: Optional[OptimizedBacktester] = None
_model_predictor: Optional[ModelPredictor] = None
_config: Optional[TradingParameters] = None
_active_backtests: Dict[str, Any] = {}


# Request/Response Models
class BacktestRequest(BaseModel):
    """Request to run a backtest."""
    tickers: List[str]
    start_date: str  # YYYY-MM-DD
    end_date: str  # YYYY-MM-DD
    starting_cash: float = 10000.0
    use_optimized: bool = True
    config_override: Optional[Dict[str, Any]] = None


class BacktestStatusResponse(BaseModel):
    """Status of a backtest."""
    backtest_id: str
    status: str  # "running", "completed", "failed"
    progress: float  # 0.0 to 1.0
    message: str
    result: Optional[Dict[str, Any]] = None


class BacktestResultResponse(BaseModel):
    """Backtest result summary."""
    backtest_id: str
    start_date: str
    end_date: str
    initial_equity: float
    final_equity: float
    total_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: Optional[float]
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    violations: int
    performance_metrics: Optional[Dict[str, Any]] = None


def create_app(config_path: Optional[str] = None) -> FastAPI:
    """Create the FastAPI application."""
    app = FastAPI(
        title="TraderV5 API",
        version="5.0.0",
        description="Advanced trading system with backtesting capabilities"
    )
    
    # CORS configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost",
            "http://localhost:19006",
            "http://127.0.0.1:19006",
            "http://localhost:8081",
            "http://localhost:3000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize on startup
    @app.on_event("startup")
    async def startup_event():
        """Initialize the trading system."""
        global _config, _backtester, _optimized_backtester, _model_predictor
        
        _LOG.info("Initializing TraderV5 API...")
        
        try:
            # Load configuration
            _config = load_trading_parameters(config_path)
            _LOG.info("Configuration loaded successfully")
            
            # Load model
            try:
                _model_predictor = ModelPredictor.load_default()
                _LOG.info("Model predictor loaded successfully")
            except Exception as e:
                _LOG.warning(f"Could not load model predictor: {e}")
                _model_predictor = None
            
            # Initialize backtester
            _backtester = CashAccountBacktester(_config)
            _LOG.info("Standard backtester initialized")
            
            # Initialize optimized backtester
            _optimized_backtester = OptimizedBacktester(config_path)
            _LOG.info("Optimized backtester initialized")
            
            _LOG.info("TraderV5 API startup complete")
            
        except Exception as e:
            _LOG.error(f"Startup failed: {e}", exc_info=True)
            raise
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """Cleanup on shutdown."""
        global _optimized_backtester
        
        _LOG.info("Shutting down TraderV5 API...")
        
        if _optimized_backtester:
            _optimized_backtester.shutdown()
        
        _LOG.info("TraderV5 API shutdown complete")
    
    # API Routes
    
    @app.get("/")
    async def index() -> Dict[str, Any]:
        """API index."""
        return {
            "name": "TraderV5 API",
            "version": "5.0.0",
            "status": "running",
            "endpoints": {
                "health": "/health",
                "config": "/config",
                "model": "/model/status",
                "backtest": {
                    "run": "POST /backtest/run",
                    "status": "GET /backtest/status/{backtest_id}",
                    "list": "GET /backtest/list",
                    "result": "GET /backtest/result/{backtest_id}",
                },
            },
        }
    
    @app.get("/health")
    async def health_check() -> Dict[str, Any]:
        """Health check endpoint."""
        return {
            "status": "healthy",
            "config_loaded": _config is not None,
            "model_loaded": _model_predictor is not None,
            "backtester_ready": _backtester is not None,
            "optimized_backtester_ready": _optimized_backtester is not None,
        }
    
    @app.get("/config")
    async def get_config() -> Dict[str, Any]:
        """Get current configuration."""
        if _config is None:
            raise HTTPException(status_code=500, detail="Configuration not loaded")
        
        return {
            "account": {
                "starting_equity": float(_config.account.starting_equity),
                "account_type": _config.account.account_type,
            },
            "risk": {
                "risk_per_trade_pct": _config.risk.risk_per_trade_pct,
                "max_concurrent_positions": _config.risk.max_concurrent_positions,
            },
            "training": {
                "tickers": _config.training.tickers,
                "lookback_days": _config.training.lookback_days,
            },
            "calendar": {
                "exchange": _config.calendar.exchange,
                "timezone": _config.calendar.timezone,
            },
        }
    
    @app.get("/model/status")
    async def model_status() -> Dict[str, Any]:
        """Get model status."""
        if _model_predictor is None:
            return {
                "loaded": False,
                "message": "Model not loaded",
            }
        
        return {
            "loaded": True,
            "model_type": type(_model_predictor).__name__,
            "ready": True,
        }
    
    @app.post("/backtest/run")
    async def run_backtest(
        request: BacktestRequest,
        background_tasks: BackgroundTasks,
    ) -> BacktestStatusResponse:
        """Run a backtest."""
        if _config is None:
            raise HTTPException(status_code=500, detail="System not initialized")
        
        if request.use_optimized and _optimized_backtester is None:
            raise HTTPException(status_code=500, detail="Optimized backtester not available")
        
        if not request.use_optimized and _backtester is None:
            raise HTTPException(status_code=500, detail="Standard backtester not available")
        
        # Generate backtest ID
        backtest_id = f"bt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Parse dates
        try:
            start_date = datetime.strptime(request.start_date, "%Y-%m-%d")
            end_date = datetime.strptime(request.end_date, "%Y-%m-%d")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid date format: {e}")
        
        # Initialize backtest status
        _active_backtests[backtest_id] = {
            "status": "running",
            "progress": 0.0,
            "message": "Initializing backtest...",
            "result": None,
        }
        
        # Run backtest in background
        background_tasks.add_task(
            _run_backtest_task,
            backtest_id,
            request.tickers,
            start_date,
            end_date,
            request.starting_cash,
            request.use_optimized,
        )
        
        return BacktestStatusResponse(
            backtest_id=backtest_id,
            status="running",
            progress=0.0,
            message="Backtest started",
        )
    
    @app.get("/backtest/status/{backtest_id}")
    async def get_backtest_status(backtest_id: str) -> BacktestStatusResponse:
        """Get backtest status."""
        if backtest_id not in _active_backtests:
            raise HTTPException(status_code=404, detail="Backtest not found")
        
        status_data = _active_backtests[backtest_id]
        
        return BacktestStatusResponse(
            backtest_id=backtest_id,
            status=status_data["status"],
            progress=status_data["progress"],
            message=status_data["message"],
            result=status_data.get("result"),
        )
    
    @app.get("/backtest/list")
    async def list_backtests() -> Dict[str, Any]:
        """List all backtests."""
        return {
            "backtests": [
                {
                    "backtest_id": backtest_id,
                    "status": data["status"],
                    "message": data["message"],
                }
                for backtest_id, data in _active_backtests.items()
            ],
            "total": len(_active_backtests),
        }
    
    @app.get("/backtest/result/{backtest_id}")
    async def get_backtest_result(backtest_id: str) -> BacktestResultResponse:
        """Get detailed backtest result."""
        if backtest_id not in _active_backtests:
            raise HTTPException(status_code=404, detail="Backtest not found")
        
        status_data = _active_backtests[backtest_id]
        
        if status_data["status"] != "completed":
            raise HTTPException(
                status_code=400,
                detail=f"Backtest not completed (status: {status_data['status']})",
            )
        
        result = status_data.get("result")
        if result is None:
            raise HTTPException(status_code=500, detail="Result not available")
        
        return BacktestResultResponse(**result)
    
    return app


async def _run_backtest_task(
    backtest_id: str,
    tickers: List[str],
    start_date: datetime,
    end_date: datetime,
    starting_cash: float,
    use_optimized: bool,
) -> None:
    """Background task to run a backtest."""
    try:
        _active_backtests[backtest_id]["message"] = "Fetching historical data..."
        _active_backtests[backtest_id]["progress"] = 0.1
        
        if use_optimized and _optimized_backtester:
            # Use optimized backtester
            _active_backtests[backtest_id]["message"] = "Running optimized backtest..."
            _active_backtests[backtest_id]["progress"] = 0.3
            
            result = _optimized_backtester.run_optimized_backtest(
                tickers=tickers,
                start_date=start_date,
                end_date=end_date,
                model_predictor=_model_predictor,
            )
        else:
            # Use standard backtester (would need to implement data fetching)
            raise NotImplementedError("Standard backtester integration pending")
        
        _active_backtests[backtest_id]["message"] = "Backtest completed"
        _active_backtests[backtest_id]["progress"] = 1.0
        _active_backtests[backtest_id]["status"] = "completed"
        
        # Convert result to dictionary
        _active_backtests[backtest_id]["result"] = {
            "backtest_id": backtest_id,
            "start_date": result.start_date.isoformat(),
            "end_date": result.end_date.isoformat(),
            "initial_equity": float(result.initial_equity),
            "final_equity": float(result.final_equity),
            "total_return_pct": float(result.total_return_pct),
            "max_drawdown_pct": float(result.max_drawdown_pct),
            "sharpe_ratio": result.sharpe_ratio,
            "total_trades": result.total_trades,
            "winning_trades": result.winning_trades,
            "losing_trades": result.losing_trades,
            "win_rate": result.win_rate,
            "violations": len(result.violations),
            "performance_metrics": getattr(result, "performance_metrics", None),
        }
        
        _LOG.info(f"Backtest {backtest_id} completed successfully")
        
    except Exception as e:
        _LOG.error(f"Backtest {backtest_id} failed: {e}", exc_info=True)
        _active_backtests[backtest_id]["status"] = "failed"
        _active_backtests[backtest_id]["message"] = f"Error: {str(e)}"
        _active_backtests[backtest_id]["progress"] = 0.0


def main():
    """Run the API server."""
    import uvicorn
    
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s - %(message)s",
    )
    
    app = create_app()
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8002,  # Different port from TraderV4
        log_level="info",
    )


if __name__ == "__main__":
    main()
