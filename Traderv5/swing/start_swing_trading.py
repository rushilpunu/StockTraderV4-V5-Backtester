#!/usr/bin/env python3
"""Launch the swing-trading tuned TraderV5 runtime.

This entry point mirrors :mod:`Traderv5.start_live_trading` but applies a risk
profile and runtime guardrails that keep activity compliant with pattern day
trading restrictions for sub-$25K accounts.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Ensure repository root is on the path so imports resolve the same way as the
# primary live trading script.
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
os.chdir(project_root)

from Traderv5.risk_profiles import SwingProfile
from Traderv5.start_live_trading import (  # noqa: E402 - import after sys.path
    check_alpaca_credentials,
    check_environment,
    ensure_models_trained,
    run_live_trading,
)

logger = logging.getLogger("traderv5.swing")

EXPECTED_EQUITY = 10_000.0
CYCLE_SECONDS = 1_800  # 30 minute cadence for slower swing checks
DAY_TRADE_LIMIT = 2  # stay comfortably inside FINRA's 3-trade rule

SWING_RISK_OVERRIDES = {
    "max_capital_fraction": SwingProfile.max_capital_fraction,
    "max_positions": SwingProfile.max_positions,
    "entry": SwingProfile.entry_threshold,
    "exit": SwingProfile.exit_threshold,
    "aggressiveness": SwingProfile.aggressiveness,
    "bias": SwingProfile.entry_bias,
    "leverage": SwingProfile.leverage,
    "cooldown": SwingProfile.cooldown,
    "allow_shorting": False,
}


def main() -> int:
    logger.info("🌙 TraderV5 Swing Trading Startup")
    logger.info("=" * 50)

    if not check_environment():
        logger.error("Environment validation failed")
        return 1

    if not check_alpaca_credentials():
        logger.error("Alpaca credential check failed")
        return 1

    predictor = ensure_models_trained(variant="swing")
    if predictor is None:
        logger.error("Unable to load trained model artifacts")
        return 1

    logger.info("🚦 Launching swing trading runtime with PDT guardrails")
    logger.info(
        "Profile: %s | Expected equity: $%s | Cycle: %d min",
        SwingProfile.name,
        f"{EXPECTED_EQUITY:,.0f}",
        CYCLE_SECONDS // 60,
    )

    try:
        run_live_trading(
            predictor,
            profile_name=SwingProfile.name,
            risk_overrides=SWING_RISK_OVERRIDES,
            cycle_pause_seconds=CYCLE_SECONDS,
            allow_shorting=False,
            day_trade_limit=DAY_TRADE_LIMIT,
            expected_equity=EXPECTED_EQUITY,
            verbose=False,
        )
        return 0
    except KeyboardInterrupt:
        logger.info("Swing trading stopped by user")
        return 0
    except Exception as exc:  # pragma: no cover - runtime guard
        logger.error("💥 Swing trading run failed: %s", exc)
        logger.exception("Swing trading failure")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
