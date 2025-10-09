#!/usr/bin/env python3
"""Complete startup script for TraderV5 live trading with Alpaca."""

import logging
import os
import sys
import signal
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Mapping, Optional

# Add parent directory to path and change to it
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))
os.chdir(parent_dir)

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _interval_to_seconds(interval: str, fallback: int = 300) -> int:
    text = (interval or "").strip().lower()
    if not text:
        return fallback
    try:
        if text.endswith("ms"):
            value = float(text[:-2]) / 1000.0
        elif text.endswith("s"):
            value = float(text[:-1])
        elif text.endswith("m"):
            value = float(text[:-1]) * 60.0
        elif text.endswith("h"):
            value = float(text[:-1]) * 3600.0
        elif text.endswith("d"):
            value = float(text[:-1]) * 86400.0
        else:
            value = float(text)
    except (TypeError, ValueError):
        return fallback
    return int(max(value, 30.0))


def check_environment():
    """Check if the environment is properly set up."""
    logger.info("🔍 Checking environment setup...")
    
    # Check if we're in the right directory
    project_root = Path(__file__).parent.parent
    expected_files = [
        project_root / "config" / "trader_v5.json",
        project_root / "config" / "credentials.py",
        project_root / "Traderv5" / "model" / "training.py"
    ]
    
    for file_path in expected_files:
        if not file_path.exists():
            logger.error(f"❌ Required file not found: {file_path}")
            return False
    
    logger.info("✅ Environment setup looks good")
    return True


def check_alpaca_credentials():
    """Check if Alpaca credentials are configured."""
    logger.info("🔑 Checking Alpaca credentials...")
    
    try:
        from config.credentials import load_alpaca_credentials
        creds = load_alpaca_credentials(required=False)
        
        if creds is None:
            logger.error("❌ Alpaca credentials not found!")
            logger.error("Please set the following environment variables:")
            logger.error("  export ALPACA_API_KEY='your_api_key'")
            logger.error("  export ALPACA_API_SECRET='your_secret_key'")
            logger.error("  export ALPACA_API_BASE='https://paper-api.alpaca.markets'  # For paper trading")
            return False
        
        logger.info("✅ Alpaca credentials found")
        logger.info(f"   Base URL: {creds.base_url}")
        logger.info(f"   API Key: {creds.api_key[:8]}...")
        return True
        
    except Exception as exc:
        logger.error(f"❌ Error checking credentials: {exc}")
        return False


def ensure_models_trained(variant: str = "core"):
    """Ensure models are trained and show model info."""

    label = variant.lower().strip()
    logger.info("🤖 Checking model training status (%s variant)...", label)

    if label in {"swing", "swing-trader", "swing_trader"}:
        try:
            from Traderv5.swing.model import load_swing_predictor
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("❌ Failed to initialise swing predictor: %s", exc)
            return None

        predictor = load_swing_predictor()
        logger.info("✅ Loaded swing trading heuristic model")
        logger.info("   🏷️  Model Variant: swing-heuristic")
        logger.info("   🎯 Horizon: multi-day (≈5 sessions)")
        logger.info("   📈 Bias: long-only, trend + sentiment confirmation")
        return predictor

    try:
        from Traderv5.model.predictor import ModelPredictor
        import joblib
        from pathlib import Path

        predictor = ModelPredictor.load_default()

        # Get model info
        model_path = Path(__file__).parent.parent / "Traderv5" / "model" / "artifacts"
        model_file = model_path / "trade_classifier.joblib"
        meta_file = model_path / "trade_classifier_meta.json"

        logger.info("✅ Models loaded successfully")
        logger.info(f"   📁 Model Path: {model_file}")

        # Show model metadata if available
        if meta_file.exists():
            import json
            with open(meta_file, 'r') as f:
                meta = json.load(f)

            logger.info(f"   🏷️  Model Type: {meta.get('model_type', 'Unknown')}")
            logger.info(f"   📊 Features: {len(meta.get('features', []))} features")
            logger.info(f"   🎯 Training Samples: {meta.get('n_samples', 'Unknown')}")

            if 'metrics' in meta:
                metrics = meta['metrics']
                logger.info(f"   📈 Model Performance:")
                if 'accuracy' in metrics:
                    logger.info(f"      • Accuracy: {metrics['accuracy']:.2%}")
                if 'precision' in metrics:
                    logger.info(f"      • Precision: {metrics['precision']:.2%}")
                if 'recall' in metrics:
                    logger.info(f"      • Recall: {metrics['recall']:.2%}")
                if 'f1_score' in metrics:
                    logger.info(f"      • F1 Score: {metrics['f1_score']:.2%}")

        # Get model file modification time
        import os
        if os.path.exists(model_file):
            import datetime
            mod_time = os.path.getmtime(model_file)
            mod_date = datetime.datetime.fromtimestamp(mod_time)
            logger.info(f"   📅 Last Updated: {mod_date.strftime('%Y-%m-%d %H:%M:%S')}")

        return predictor

    except FileNotFoundError:
        logger.error("❌ No trained models found")
        logger.error("Run: python3 Traderv5/model/training.py")
        return None
    except Exception as exc:
        logger.error(f"❌ Error loading models: {exc}")
        import traceback
        traceback.print_exc()
        return None


def run_live_trading(
    predictor,
    *,
    profile_name: Optional[str] = None,
    risk_overrides: Optional[Mapping[str, object]] = None,
    cycle_pause_seconds: Optional[int] = None,
    allow_shorting: Optional[bool] = None,
    day_trade_limit: Optional[int] = None,
    expected_equity: Optional[float] = None,
    verbose: bool = True,
):
    """Start the live trading system."""
    logger.info("🚀 Starting live trading system...")
    
    shutdown_requested = False
    
    def signal_handler(signum, frame):
        nonlocal shutdown_requested
        logger.info(f"Received signal {signum}, shutting down...")
        shutdown_requested = True
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        from Traderv5.configuration import load_trading_parameters
        from Traderv5.trader import ModelDrivenTrader, TraderV5Config
        from Traderv4.funcs import RiskConfig
        from Traderv5.risk_profiles import AggressiveProfile, apply_profile, get_profile
        from Traderv5.persistence import PersistentTradeJournal
        from config.credentials import load_alpaca_credentials
        from alpaca_trade_api import REST

        # Load config
        logger.info("Loading configuration...")
        config_params = load_trading_parameters()

        # Create TraderV5Config
        env_profile = os.getenv("TRADERV5_RISK_PROFILE")
        requested_profile = profile_name or env_profile or AggressiveProfile.name
        try:
            profile = get_profile(requested_profile)
        except KeyError:
            logger.warning(
                "Unknown risk profile '%s'; falling back to '%s'",
                requested_profile,
                AggressiveProfile.name,
            )
            profile = AggressiveProfile
        else:
            if requested_profile.lower() != profile.name:
                logger.warning(
                    "Normalised live risk profile from '%s' to '%s'", requested_profile, profile.name
                )
            elif profile_name and env_profile and profile_name.lower() != env_profile.lower():
                logger.info(
                    "Overriding TRADERV5_RISK_PROFILE='%s' with explicit profile '%s'",
                    env_profile,
                    profile.name,
                )

        logger.info("Applying %s risk profile", profile.name.title())

        risk_config = RiskConfig(
            max_capital_fraction=profile.max_capital_fraction,
            max_positions=profile.max_positions,
            stop_loss_pct=config_params.risk.atr_stop_multiplier * 0.01,
            take_profit_pct=config_params.risk.take_profit_multiple * 0.01,
            cooldown_minutes=profile.cooldown,
            entry_sentiment_threshold=profile.entry_threshold,
            exit_sentiment_threshold=profile.exit_threshold,
            allow_shorting=True,
        )

        capital_from_risk = max(
            profile.max_capital_fraction,
            config_params.risk.risk_per_trade_pct * 8.5,
        )
        overrides = {
            "max_capital_fraction": min(0.48, capital_from_risk),
            "max_positions": max(profile.max_positions, 6),
            "entry": max(0.032, profile.entry_threshold * 0.82),
            "exit": max(0.015, profile.exit_threshold * 0.78),
            "aggressiveness": profile.aggressiveness * max(1.05, config_params.risk.aggressiveness),
            "bias": profile.entry_bias + max(0.0, config_params.risk.entry_signal_bias),
            "leverage": max(profile.leverage, config_params.risk.max_trade_leverage or 1.6),
        }
        if risk_overrides:
            logger.info(
                "Applying %d custom risk override(s): %s",
                len(risk_overrides),
                ", ".join(sorted(risk_overrides.keys())),
            )
            overrides.update(risk_overrides)
        if allow_shorting is not None:
            overrides["allow_shorting"] = allow_shorting
        else:
            overrides.setdefault("allow_shorting", True)

        apply_profile(risk_config, profile, overrides=overrides)

        logger.info(
            "Risk settings → max_capital_fraction=%.3f | max_positions=%d | cooldown=%d min | leverage=%.2f | shorting=%s",
            risk_config.max_capital_fraction,
            risk_config.max_positions,
            risk_config.cooldown_minutes,
            getattr(risk_config, "max_trade_leverage", 1.0),
            "yes" if risk_config.allow_shorting else "no",
        )
        logger.info(
            "Entry/exit thresholds → entry=%.3f | exit=%.3f | aggressiveness=%.2f | bias=%.3f",
            risk_config.entry_sentiment_threshold,
            risk_config.exit_sentiment_threshold,
            getattr(risk_config, "aggressiveness", 1.0),
            getattr(risk_config, "entry_signal_bias", 0.0),
        )

        cycle_interval = cycle_pause_seconds if cycle_pause_seconds and cycle_pause_seconds > 0 else 300

        trader_config = TraderV5Config(
            tickers=list(config_params.training.tickers),
            lookback_days=config_params.training.lookback_days,
            price_interval=config_params.training.price_interval,
            gdelt_timeline_minutes=config_params.training.timeline_minutes,
            gdelt_delay=config_params.data.gdelt_pause_seconds,
            sentiment_window_minutes=config_params.training.timeline_minutes,
            cycle_pause_seconds=cycle_interval,
            risk=risk_config,
            label_horizon_minutes=config_params.training.label_horizon_minutes,
            positive_threshold=config_params.training.positive_threshold,
            negative_threshold=config_params.training.negative_threshold,
        )
        
        logger.info(f"Trading {len(trader_config.tickers)} tickers: {', '.join(trader_config.tickers)}")
        
        # Connect to Alpaca
        logger.info("Connecting to Alpaca...")
        creds = load_alpaca_credentials()
        alpaca_client = REST(
            key_id=creds.api_key,
            secret_key=creds.api_secret,
            base_url=creds.base_url
        )
        
        # Verify connection
        account = alpaca_client.get_account()
        logger.info(f"Connected to Alpaca account: {account.account_number}")
        equity_value = float(account.equity)
        cash_value = float(account.cash)
        logger.info(f"Equity: ${equity_value:,.2f}")
        logger.info(f"Cash: ${cash_value:,.2f}")
        if expected_equity is not None:
            try:
                expected_equity_value = float(expected_equity)
            except (TypeError, ValueError):
                logger.warning("Invalid expected_equity=%r provided; skipping comparison", expected_equity)
            else:
                delta = equity_value - expected_equity_value
                logger.info(
                    f"Target equity baseline: ${expected_equity_value:,.2f} (Δ {delta:+,.2f})",
                )

        clock = alpaca_client.get_clock()
        logger.info(f"Market is {'OPEN' if clock.is_open else 'CLOSED'}")
        
        # Create trader with verbose logging
        journal_filename = f"traderv5_{profile.name}_journal.json"
        journal_path = Path("data/state") / journal_filename
        trade_journal = PersistentTradeJournal(risk_config, storage_path=journal_path, variant=profile.name)

        class VerboseTrader(ModelDrivenTrader):
            def run_cycle(self, tickers: Optional[List[str]] = None) -> None:
                batch = ", ".join(tickers or self.config.tickers)
                logger.info("🔄 Evaluating tickers: %s", batch)
                super().run_cycle(tickers)
                logger.info("✅ Cycle complete for %s", batch)

        trader_cls = VerboseTrader if verbose else ModelDrivenTrader
        trader = trader_cls(
            config=trader_config,
            alpaca_client=alpaca_client,
            balance_fetcher=lambda: float(alpaca_client.get_account().cash),
            predictor=predictor,
            trade_memory=trade_journal,
        )

        if day_trade_limit is not None:
            try:
                enforced_limit = max(int(day_trade_limit), 1)
            except (TypeError, ValueError):
                logger.warning(
                    "Invalid day_trade_limit=%r provided; defaulting to 3",
                    day_trade_limit,
                )
                enforced_limit = 3
            trader.trade_tracker.max_day_trades = enforced_limit
            logger.info(
                "Pattern day trading guard active: max %d intraday round-trips per rolling 5 trading days",
                enforced_limit,
            )

        price_interval_seconds = _interval_to_seconds(trader_config.price_interval, cycle_interval)
        base_interval = max(60, min(price_interval_seconds, cycle_interval))
        active_interval = max(45, base_interval // 2)

        logger.info("✅ TraderV5 initialized successfully!")
        logger.info("")
        logger.info("=" * 80)
        logger.info("🎯 LIVE TRADING MODE ACTIVE")
        logger.info("=" * 80)
        logger.info(f"📊 Trading Schedule:")
        if base_interval >= 60:
            logger.info(f"   • Base Evaluation Interval: {base_interval // 60} minutes")
        else:
            logger.info(f"   • Base Evaluation Interval: {base_interval} seconds")
        if active_interval >= 60:
            logger.info(f"   • Active Position Check: every {active_interval // 60} minutes")
        else:
            logger.info(f"   • Active Position Check: every {active_interval} seconds")
        logger.info(f"   • Trading Tickers: {', '.join(trader_config.tickers)}")
        logger.info(f"   • Lookback Period: {trader_config.lookback_days} days")
        logger.info("")
        logger.info("💡 The system reacts to:")
        logger.info("   1. New price bars from Yahoo Finance")
        logger.info("   2. Open positions with stored exit plans")
        if verbose:
            logger.info("   3. Verbose per-ticker diagnostics")
        logger.info("")
        logger.info("Press Ctrl+C to stop gracefully")
        logger.info("=" * 80)
        logger.info("")

        status_interval = 45
        last_status_time = datetime.utcnow()

        while not shutdown_requested:
            try:
                now = datetime.utcnow()
                if trade_journal is not None:
                    due, next_wait = trade_journal.schedule(
                        trader_config.tickers,
                        now=now,
                        base_interval_seconds=base_interval,
                        active_interval_seconds=active_interval,
                    )
                    open_count = trade_journal.active_positions()
                else:
                    due = list(trader_config.tickers)
                    next_wait = base_interval
                    open_count = 0

                if due:
                    logger.debug("Triggering evaluation for %s", ", ".join(due))
                    trader.run_cycle(due)
                    last_status_time = now
                    continue

                if (now - last_status_time).total_seconds() >= status_interval:
                    logger.info("⏳ Waiting %.0fs for next evaluation | active plans: %d", next_wait, open_count)
                    last_status_time = now

                sleep_for = min(max(int(next_wait), 20), 600)
                time.sleep(sleep_for)
            except Exception as exc:
                logger.error(f"Error in trading loop: {exc}")
                logger.exception("Runtime loop error")
                time.sleep(60)

        logger.info("Trading stopped")
        if trade_journal is not None:
            trade_journal.flush()
        
    except KeyboardInterrupt:
        logger.info("🛑 Trading stopped by user")
    except Exception as exc:
        logger.error(f"❌ Error in trading system: {exc}")
        logger.exception("Trading error")
        raise


def main():
    """Main startup sequence."""
    logger.info("🎯 TraderV5 Live Trading Startup")
    logger.info("=" * 50)
    
    # Step 1: Check environment
    if not check_environment():
        logger.error("💥 Environment check failed")
        return 1
    
    # Step 2: Check Alpaca credentials
    if not check_alpaca_credentials():
        logger.error("💥 Credentials check failed")
        return 1
    
    # Step 3: Ensure models are trained
    predictor = ensure_models_trained()
    if predictor is None:
        logger.error("💥 Model loading failed")
        return 1
    
    # Step 4: Start trading
    logger.info("🎉 All checks passed! Starting live trading...")
    logger.info("-" * 50)
    
    try:
        run_live_trading(predictor)
        return 0
    except Exception as exc:
        logger.error(f"💥 Trading failed: {exc}")
        logger.exception("Trading failure")
        return 1


if __name__ == "__main__":
    exit(main())
