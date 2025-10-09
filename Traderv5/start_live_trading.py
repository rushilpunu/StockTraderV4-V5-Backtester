#!/usr/bin/env python3
"""Complete startup script for TraderV5 live trading with Alpaca."""

import logging
import os
import sys
import signal
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Mapping, Optional

from Traderv5.persistence import PositionPersistence

# Add parent directory to path and change to it
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
os.chdir(parent_dir)

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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
    position_store_name: Optional[str] = None,
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

        store_variant = (position_store_name or profile.name or AggressiveProfile.name)
        position_store = PositionPersistence.for_variant(store_variant)
        logger.info("Persistent position memory initialised at %s", position_store.path)
        
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
        logger.info("Initializing trader...")
        
        # Wrap the trader to add detailed decision logging
        class VerboseTrader(ModelDrivenTrader):
            def _run_cycle_impl(self):
                """Override to add detailed logging."""
                from Traderv5.data_sources import YahooFinanceDataFetcher, collect_gdelt_window
                import pandas as pd
                
                cycle_started = datetime.utcnow()
                logger.info("")
                logger.info("=" * 80)
                logger.info(f"🔄 TRADING CYCLE STARTED: {cycle_started.strftime('%Y-%m-%d %H:%M:%S UTC')}")
                logger.info("=" * 80)
                
                # Get account info
                account_snapshot = self.executor.get_account_snapshot()
                start_cash = account_snapshot.cash
                start_equity = account_snapshot.equity
                logger.info(f"💰 Account: Cash=${start_cash:,.2f} | Equity=${start_equity:,.2f}")
                
                # Get positions
                positions = self.executor.get_open_positions()
                logger.info(f"📊 Open Positions: {len(positions)}")
                for pos in positions:
                    symbol = pos.get('ticker', 'UNKNOWN')
                    qty = pos.get('qty', 0)
                    market_val = pos.get('market_value', 0)
                    logger.info(f"   • {symbol}: {qty} shares (${market_val:,.2f})")
                
                # Get market status
                market_clock = self.executor.get_market_clock()
                is_open = market_clock.get("is_open", False) if market_clock else False
                logger.info(f"🏦 Market Status: {'OPEN ✅' if is_open else 'CLOSED ❌'}")
                logger.info("")
                
                if not is_open:
                    logger.info("⏸️  Market is closed, skipping trading")
                    logger.info("")
                    return
                
                # Track actions taken and skipped tickers
                actions_taken = []
                skipped_tickers = []
                
                # Process each ticker with detailed logging
                from datetime import timezone, timedelta
                end = datetime.utcnow().replace(tzinfo=timezone.utc)
                # Live sentiment window: limit to last 24h to reduce API load
                start = end - timedelta(hours=24)
                
                # FETCH GDELT DATA - Use intelligent multi-term queries for each ticker
                logger.info("📰 Fetching GDELT data...")
                gdelt_start_time = datetime.utcnow()
                gdelt_data = {}
                
                # Build intelligent query terms per ticker
                def _terms_for(t: str) -> list[str]:
                    t = t.upper()
                    if t == "AAPL":
                        return ["Apple", "iPhone", "iPad", "Mac", "MacBook", "Apple Watch", "Apple Vision Pro"]
                    if t == "MSFT":
                        return ["Microsoft", "Windows", "Azure", "Surface", "Xbox", "Copilot", "OpenAI"]
                    if t == "AMZN":
                        return ["Amazon", "AWS", "Kindle", "Prime", "Whole Foods"]
                    if t == "NVDA":
                        return ["NVIDIA", "GeForce", "CUDA", "DGX", "AI chips", "H100", "GPU"]
                    if t == "TSLA":
                        return ["Tesla", "Elon Musk", "Model 3", "Model Y", "Cybertruck", "Gigafactory", "FSD"]
                    if t == "META":
                        return ["Meta", "Facebook", "Instagram", "WhatsApp", "Quest", "Reality Labs"]
                    if t == "GOOGL" or t == "GOOG":
                        return ["Google", "Alphabet", "Android", "YouTube", "Gemini", "Search"]
                    if t == "AMD":
                        return ["AMD", "Ryzen", "EPYC", "Instinct", "AI chips"]
                    return []
                
                from Traderv5.services.gdelt import GDELTService
                from Traderv5.http_client import RateLimitedHttpClient
                from Traderv5.configuration import DataSettings, CalendarSettings
                from datetime import timedelta as _td
                http = RateLimitedHttpClient(
                    queries_per_second=0.1,  # slow down to avoid 429s
                    cache_dir=Path("cache/http/gdelt"),
                    cache_ttl=_td(hours=6),
                    max_retries=6,
                    backoff_factor=3,
                )
                # Heavier chunk size + per-chunk pause to reduce request count
                gdelt_data_settings = DataSettings(
                    queries_per_second=0.25,
                    max_retries=6,
                    backoff_factor=3.0,
                    cache_ttl_hours=6.0,
                    cache_dir=Path("cache/http/gdelt"),
                    gdelt_chunk_minutes=720,   # 12h chunks
                    gdelt_pause_seconds=5.5,   # pause between chunks
                    yahoo_pause_seconds=0.8,
                )
                gdelt_service = GDELTService(http, calendar_settings=CalendarSettings(), data_settings=gdelt_data_settings)
                
                for ticker in self.config.tickers:
                    try:
                        window = gdelt_service.fetch_sentiment_window(
                            ticker,
                            start.replace(tzinfo=None),
                            end.replace(tzinfo=None),
                            query_terms=_terms_for(ticker),
                        )
                        gdelt_data[ticker] = window
                        logger.info(f"   ✅ {ticker}: {len(window.articles)} articles, {sum(s.article_count for s in window.summaries)} daily")
                    except Exception as exc:
                        logger.warning(f"   ⚠️  {ticker}: GDELT failed, skipping ticker - {str(exc)[:50]}")
                        gdelt_data[ticker] = None
                    # Small pause between tickers to reduce burstiness
                    time.sleep(5.5)
                
                logger.info(f"   ⏱️  Done in {(datetime.utcnow() - gdelt_start_time).total_seconds():.1f}s")
                logger.info("")
                
                # Now process each ticker with pre-fetched GDELT data
                for ticker in self.config.tickers:
                    # Skip ticker if GDELT data fetch failed
                    if gdelt_data.get(ticker) is None:
                        logger.info("-" * 80)
                        logger.info(f"⏭️  Skipping {ticker} - GDELT data unavailable")
                        logger.info("-" * 80)
                        logger.info("")
                        skipped_tickers.append(ticker)
                        continue
                    
                    logger.info("-" * 80)
                    logger.info(f"🎯 Analyzing: {ticker}")
                    logger.info("-" * 80)
                    
                    try:
                        # 1. Fetch price data from Yahoo Finance
                        logger.info(f"📈 Fetching Yahoo Finance data...")
                        price_df = self._fetch_price_window(ticker, start, end)
                        
                        if price_df.empty:
                            logger.info(f"❌ {ticker}: No price data available")
                            continue
                        
                        current_price = float(price_df["close"].iloc[-1])
                        volume = float(price_df["volume"].iloc[-1]) if "volume" in price_df else 0
                        price_change = ((current_price - float(price_df["close"].iloc[0])) / float(price_df["close"].iloc[0])) * 100
                        
                        logger.info(f"   💵 Price: ${current_price:.2f} ({price_change:+.2f}% over {len(price_df)} periods)")
                        logger.info(f"   📊 Volume: {volume:,.0f}")
                        
                        # 2. Use pre-fetched GDELT sentiment data
                        logger.info(f"📰 Using GDELT sentiment data...")
                        gdelt_window = gdelt_data.get(ticker)

                        # GDELT data is guaranteed to exist (we filtered out None above)
                        articles = gdelt_window.articles or []
                        article_count = len(articles)
                        timeline_points = sum(summary.timeline_points for summary in gdelt_window.summaries)

                        # Prefer article-level tone when the field is present, otherwise fall back to timeline summaries
                        article_tones = [article.tone for article in articles if article.raw.get("tone") is not None]
                        tone_source = "articles" if article_tones else "timeline"
                        if article_tones:
                            avg_sentiment = sum(article_tones) / max(len(article_tones), 1)
                        else:
                            weighted_pairs = []
                            for summary in gdelt_window.summaries:
                                weight = summary.article_count + summary.timeline_points
                                weighted_pairs.append((summary.average_tone, max(weight, 1)))
                            if weighted_pairs:
                                numerator = sum(value * weight for value, weight in weighted_pairs)
                                denominator = sum(weight for _, weight in weighted_pairs)
                                avg_sentiment = numerator / max(denominator, 1)
                            else:
                                avg_sentiment = 0.0

                        logger.info(f"   📰 Articles: {article_count}")
                        if tone_source == "articles" and article_count > len(article_tones):
                            logger.info("   ⚠️ Some articles are missing tone data in the GDELT response")
                        if tone_source == "timeline" and article_count > 0:
                            logger.info("   ℹ️ GDELT omitted per-article tone; falling back to timeline averages")
                        logger.info(f"   ⏱️  Timeline Points: {timeline_points}")
                        logger.info(f"   😊 Avg Sentiment ({tone_source}): {avg_sentiment:.3f}")
                        
                        # 3. Build features
                        logger.info(f"🔧 Building ML features...")
                        # Convert our simple GDELT data to the format expected by feature_engineer
                        # The feature engineer expects article objects with proper structure
                        feature_frame = self.feature_engineer.build_training_frame(
                            ticker,
                            price_df,
                            gdelt_window,
                            include_labels=False,
                        )
                        
                        if feature_frame.empty:
                            logger.info(f"❌ {ticker}: No features available")
                            continue
                        
                        latest_row = feature_frame.iloc[-1]
                        feature_columns = self.feature_engineer.feature_columns(feature_frame)
                        features = latest_row[feature_columns].to_dict()
                        
                        # 4. Get model prediction
                        logger.info(f"🤖 Running ML model prediction...")
                        probability = float(self.predictor.predict_proba(features))
                        signal_strength = probability - 0.5
                        
                        logger.info(f"   🎲 Probability (Long): {probability:.1%}")
                        logger.info(f"   📊 Signal Strength: {signal_strength:+.3f}")
                        
                        # 5. Decision breakdown
                        logger.info(f"")
                        logger.info(f"⚖️  DECISION BREAKDOWN:")
                        logger.info(f"   📰 GDELT Sentiment: {avg_sentiment:.3f} → {'BULLISH ✅' if avg_sentiment > 0 else 'BEARISH ❌'}")
                        logger.info(f"   📈 Yahoo Price Trend: {price_change:+.2f}% → {'UP ✅' if price_change > 0 else 'DOWN ❌'}")
                        logger.info(f"   🤖 Model Prediction: {probability:.1%} → {'BUY ✅' if probability > 0.6 else 'SELL ❌' if probability < 0.4 else 'HOLD ⏸️'}")
                        logger.info(f"")
                        
                        # Show feature importance (top 5)
                        feature_values = [(k, v) for k, v in features.items() if isinstance(v, (int, float))]
                        feature_values.sort(key=lambda x: abs(x[1]), reverse=True)
                        
                        logger.info(f"   🔍 Top Features:")
                        for feat_name, feat_value in feature_values[:5]:
                            logger.info(f"      • {feat_name}: {feat_value:.3f}")
                        
                        # 6. Make decision
                        price_snapshot = {"close": current_price, "volume": volume}
                        position_ctx = next((p for p in positions if p.get("ticker", "").upper() == ticker), None)
                        
                        decision = self.decision_engine.decide(
                            ticker,
                            features=features,
                            price_snapshot=price_snapshot,
                            account=account_snapshot,
                            position=position_ctx,
                            open_positions=len(positions),
                        )
                        
                        # 7. Log final decision
                        logger.info(f"")
                        logger.info(f"📋 FINAL DECISION:")
                        logger.info(f"   Action: {decision.action}")
                        logger.info(f"   Reason: {decision.reason}")
                        
                        if decision.action != "HOLD":
                            logger.info(f"   Quantity: {decision.quantity} shares")
                            logger.info(f"   Notional: ${decision.notional:,.2f}")
                            if decision.stop_loss:
                                logger.info(f"   Stop Loss: ${decision.stop_loss:.2f}")
                            if decision.take_profit:
                                logger.info(f"   Take Profit: ${decision.take_profit:.2f}")
                            
                            # Execute trade
                            order_id = self.executor.place_trade(decision)
                            if order_id:
                                logger.info(f"   ✅ Order placed: {order_id}")
                                actions_taken.append({
                                    'ticker': ticker,
                                    'action': decision.action,
                                    'quantity': decision.quantity,
                                    'notional': decision.notional,
                                    'order_id': order_id
                                })
                            else:
                                logger.info(f"   ❌ Order failed")
                        
                        logger.info("")
                        
                    except Exception as exc:
                        logger.error(f"❌ Error processing {ticker}: {exc}")
                        import traceback
                        traceback.print_exc()
                        continue
                
                # Get final account state
                final_account = self.executor.get_account_snapshot()
                final_positions = self.executor.get_open_positions()
                
                cycle_ended = datetime.utcnow()
                duration = (cycle_ended - cycle_started).total_seconds()
                
                logger.info("=" * 80)
                logger.info(f"✅ CYCLE SUMMARY")
                logger.info("=" * 80)
                logger.info(f"⏱️  Duration: {duration:.1f}s")
                logger.info(f"💰 Account Change: Cash ${start_cash:,.2f} → ${final_account.cash:,.2f} ({final_account.cash - start_cash:+,.2f})")
                logger.info(f"📊 Equity Change: ${start_equity:,.2f} → ${final_account.equity:,.2f} ({final_account.equity - start_equity:+,.2f})")
                
                # Show skipped tickers
                if skipped_tickers:
                    logger.info(f"⏭️  Skipped Tickers: {len(skipped_tickers)} ({', '.join(skipped_tickers)})")
                
                logger.info(f"📈 Actions Taken: {len(actions_taken)}")
                
                if actions_taken:
                    for action in actions_taken:
                        logger.info(f"   • {action['action']} {action['ticker']}: {action['quantity']} shares (${action['notional']:,.2f})")
                else:
                    logger.info(f"   • No trades executed (all HOLD)")
                
                logger.info(f"🏦 Current Positions: {len(final_positions)}")
                for pos in final_positions:
                    symbol = pos.get('ticker', 'UNKNOWN')
                    qty = pos.get('qty', 0)
                    market_val = pos.get('market_value', 0)
                    logger.info(f"   • {symbol}: {qty} shares (${market_val:,.2f})")
                
                logger.info("=" * 80)
                logger.info("")
        
        if not verbose:
            logger.info("Verbose mode disabled; using streamlined cycle logging output.")

        trader_cls = VerboseTrader if verbose else ModelDrivenTrader
        trader = trader_cls(
            config=trader_config,
            alpaca_client=alpaca_client,
            balance_fetcher=lambda: float(alpaca_client.get_account().cash),
            predictor=predictor,
            position_store=position_store,
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

        logger.info("✅ TraderV5 initialized successfully!")
        logger.info("")
        logger.info("=" * 80)
        logger.info("🎯 LIVE TRADING MODE ACTIVE")
        logger.info("=" * 80)
        logger.info(f"📊 Trading Schedule:")
        logger.info(f"   • Cycle Interval: {trader_config.cycle_pause_seconds // 60} minutes")
        logger.info(f"   • Trading Tickers: {', '.join(trader_config.tickers)}")
        logger.info(f"   • Lookback Period: {trader_config.lookback_days} days")
        logger.info("")
        logger.info("💡 The system is now running. You will see:")
        logger.info("   1. Market status checks every 30 seconds")
        logger.info(
            "   2. Full trading cycles every %d minutes (when market is open)",
            max(trader_config.cycle_pause_seconds // 60, 1),
        )
        if verbose:
            logger.info("   3. Detailed analysis for each ticker")
        else:
            logger.info("   3. Summary trade execution logs per cycle")
        logger.info("")
        logger.info("Press Ctrl+C to stop gracefully")
        logger.info("=" * 80)
        logger.info("")
        
        # Trading loop
        cycle_count = 0
        last_cycle_time = datetime.utcnow() - timedelta(seconds=trader_config.cycle_pause_seconds)  # Allow first cycle immediately
        last_status_time = datetime.utcnow()
        status_interval = 30  # Show status every 30 seconds
        
        while not shutdown_requested:
            try:
                current_time = datetime.utcnow()
                time_since_last = (current_time - last_cycle_time).total_seconds()
                time_since_status = (current_time - last_status_time).total_seconds()
                
                # Show periodic status updates (but not before first cycle)
                if cycle_count > 0 and time_since_status >= status_interval:
                    next_cycle_in = max(0, trader_config.cycle_pause_seconds - time_since_last)
                    logger.info(f"⏰ [{current_time.strftime('%H:%M:%S')}] System active - Next cycle in {int(next_cycle_in)}s | Cycles completed: {cycle_count}")
                    last_status_time = current_time
                
                # Run trading cycle
                if time_since_last >= trader_config.cycle_pause_seconds:
                    cycle_count += 1
                    
                    trader.run_cycle()
                    last_cycle_time = current_time
                else:
                    time.sleep(10)  # Check every 10 seconds
                    
            except Exception as exc:
                logger.error(f"Error in trading cycle: {exc}")
                logger.exception("Cycle error")
                time.sleep(60)  # Wait before retry
        
        logger.info("Trading stopped")
        
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
