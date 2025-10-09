#!/usr/bin/env python3
"""Comprehensive backtester control script for TraderV5."""

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backtester.config import BacktestConfig
from backtester.runner import BacktestRunner
from backtester.bots.base import BOT_REGISTRY

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s - %(message)s"
)
_LOG = logging.getLogger("comprehensive_backtest")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive backtests for TraderV4/V5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run TraderV5 on AAPL for last 30 days
  python backtester/run_comprehensive_backtest.py --tickers AAPL --days 30 --bot traderv5
  
  # Compare V4 and V5 on multiple tickers
  python backtester/run_comprehensive_backtest.py --tickers AAPL MSFT GOOGL --days 60 --bot traderv4 traderv5
  
  # Run on specific date range
  python backtester/run_comprehensive_backtest.py --tickers TSLA --start 2024-01-01 --end 2024-12-31
  
  # Full year backtest with detailed output
  python backtester/run_comprehensive_backtest.py --tickers AAPL MSFT --year 2024 --trace --output results.json
        """
    )
    
    # Ticker selection
    parser.add_argument(
        "--tickers",
        nargs="+",
        required=True,
        help="Tickers to backtest (e.g., AAPL MSFT GOOGL)"
    )
    
    # Date range options
    date_group = parser.add_mutually_exclusive_group(required=True)
    date_group.add_argument(
        "--days",
        type=int,
        help="Number of days to backtest (from today backwards)"
    )
    date_group.add_argument(
        "--year",
        type=int,
        help="Backtest for entire year (e.g., 2024)"
    )
    date_group.add_argument(
        "--dates",
        nargs=2,
        metavar=("START", "END"),
        help="Specific date range (YYYY-MM-DD YYYY-MM-DD)"
    )
    
    # Trading parameters
    parser.add_argument(
        "--cash",
        type=float,
        default=10_000,
        help="Starting cash (default: 10000)"
    )
    parser.add_argument(
        "--bot",
        "--bots",
        nargs="+",
        default=["traderv5"],
        help="Bot variants to test (default: traderv5)"
    )
    
    # Data parameters
    parser.add_argument(
        "--window",
        type=int,
        default=60,
        help="Sentiment window in minutes (default: 60)"
    )
    parser.add_argument(
        "--timeframe",
        default="1Hour",
        help="Bar timeframe (default: 1Hour)"
    )
    
    # Feature flags
    parser.add_argument(
        "--no-vader",
        action="store_true",
        help="Disable VADER sentiment"
    )
    parser.add_argument(
        "--finbert",
        action="store_true",
        help="Enable FinBERT sentiment (slower)"
    )
    parser.add_argument(
        "--keybert",
        action="store_true",
        help="Enable KeyBERT keyword extraction"
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Record detailed execution trace"
    )
    
    # Performance
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of parallel workers (0=auto, default: 0)"
    )
    
    # Output
    parser.add_argument(
        "--output",
        "-o",
        help="Save results to JSON file"
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress detailed output"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Show comparison table when running multiple bots"
    )
    
    return parser.parse_args()


def calculate_date_range(args: argparse.Namespace) -> tuple[datetime, datetime]:
    """Calculate start and end dates from arguments."""
    if args.days:
        end = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        start = end - timedelta(days=args.days)
        return start, end
    
    elif args.year:
        start = datetime(args.year, 1, 1)
        end = datetime(args.year, 12, 31, 23, 59, 59)
        return start, end
    
    elif args.dates:
        start = datetime.strptime(args.dates[0], "%Y-%m-%d")
        end = datetime.strptime(args.dates[1], "%Y-%m-%d")
        return start, end
    
    raise ValueError("No date range specified")


def display_results(
    results: Dict,
    config: BacktestConfig,
    quiet: bool = False,
    compare: bool = False
) -> None:
    """Display backtest results in a formatted way."""
    if quiet:
        return
    
    print("\n" + "=" * 80)
    print("BACKTEST RESULTS")
    print("=" * 80)
    
    print(f"\nConfiguration:")
    print(f"  Tickers: {', '.join(config.tickers)}")
    print(f"  Period: {config.start.date()} to {config.end.date()}")
    print(f"  Duration: {(config.end - config.start).days} days")
    print(f"  Starting Cash: ${config.starting_cash:,.2f}")
    print(f"  Bots: {', '.join(config.bot_variants)}")
    
    if not results:
        print("\n⚠️  No results generated")
        return
    
    # Group results by bot
    bot_results: Dict[str, List] = {}
    for key, metrics in results.items():
        bot_name = key.split(":")[0] if ":" in key else key
        if bot_name not in bot_results:
            bot_results[bot_name] = []
        bot_results[bot_name].append((key, metrics))
    
    # Display results for each bot
    for bot_name, bot_data in bot_results.items():
        print(f"\n{'─' * 80}")
        print(f"BOT: {bot_name.upper()}")
        print(f"{'─' * 80}")
        
        total_return = 0.0
        total_trades = 0
        total_alerts = 0
        profitable_trades = 0
        
        for key, metrics in bot_data:
            ticker = key.split(":")[1] if ":" in key else "UNKNOWN"
            
            print(f"\n  {ticker}:")
            print(f"    Total Return: ${metrics.total_return:,.2f} ({(metrics.total_return/config.starting_cash)*100:.2f}%)")
            print(f"    Trades: {len(metrics.trades)}")
            print(f"    Alerts: {metrics.total_alerts}")
            print(f"    Profitable Trades: {metrics.profitable_trades}")
            print(f"    Win Rate: {metrics.win_rate:.1%}")
            print(f"    Alert Precision: {metrics.alert_precision:.1%}")
            print(f"    Max Drawdown: {metrics.max_drawdown:.2%}")
            
            if metrics.feature_summary:
                print(f"    Average Score: {metrics.feature_summary.get('averageScore', 0):.3f}")
                keywords = metrics.feature_summary.get('topKeywords', [])
                if keywords:
                    print(f"    Top Keywords: {', '.join(keywords[:3])}")
            
            total_return += metrics.total_return
            total_trades += len(metrics.trades)
            total_alerts += metrics.total_alerts
            profitable_trades += metrics.profitable_trades
        
        # Summary for bot
        print(f"\n  Summary:")
        print(f"    Combined Return: ${total_return:,.2f}")
        print(f"    Total Trades: {total_trades}")
        print(f"    Total Alerts: {total_alerts}")
        print(f"    Overall Win Rate: {(profitable_trades/total_trades*100 if total_trades > 0 else 0):.1f}%")
    
    # Comparison table if multiple bots
    if compare and len(bot_results) > 1:
        print(f"\n{'═' * 80}")
        print("COMPARISON")
        print(f"{'═' * 80}")
        
        comparison_data = []
        for bot_name, bot_data in bot_results.items():
            total_return = sum(metrics.total_return for _, metrics in bot_data)
            total_trades = sum(len(metrics.trades) for _, metrics in bot_data)
            profitable = sum(metrics.profitable_trades for _, metrics in bot_data)
            win_rate = (profitable / total_trades * 100) if total_trades > 0 else 0
            
            comparison_data.append({
                "Bot": bot_name,
                "Return": total_return,
                "Trades": total_trades,
                "Win Rate": win_rate,
            })
        
        # Display comparison
        print(f"\n{'Bot':<15} {'Return':<15} {'Trades':<10} {'Win Rate':<12}")
        print(f"{'-'*15} {'-'*15} {'-'*10} {'-'*12}")
        for data in comparison_data:
            print(
                f"{data['Bot']:<15} "
                f"${data['Return']:>12,.2f} "
                f"{data['Trades']:>10} "
                f"{data['Win Rate']:>10.1f}%"
            )
    
    print("\n" + "=" * 80)


def save_results_to_file(results: Dict, config: BacktestConfig, output_path: str) -> None:
    """Save results to a JSON file."""
    output_data = {
        "config": {
            "tickers": config.tickers,
            "start": config.start.isoformat(),
            "end": config.end.isoformat(),
            "starting_cash": config.starting_cash,
            "bot_variants": config.bot_variants,
        },
        "results": {},
    }
    
    for key, metrics in results.items():
        output_data["results"][key] = {
            "total_return": metrics.total_return,
            "total_trades": len(metrics.trades),
            "total_alerts": metrics.total_alerts,
            "profitable_trades": metrics.profitable_trades,
            "win_rate": metrics.win_rate,
            "alert_precision": metrics.alert_precision,
            "max_drawdown": metrics.max_drawdown,
            "pnl_curve": metrics.pnl_curve,
            "feature_summary": metrics.feature_summary,
            "debug": metrics.debug,
        }
    
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    _LOG.info(f"Results saved to {output_path}")


def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    # Show available bots
    if not args.quiet:
        available_bots = list(BOT_REGISTRY.keys())
        print(f"\n{'='*80}")
        print("COMPREHENSIVE BACKTESTER")
        print(f"{'='*80}")
        print(f"Available bots: {', '.join(available_bots)}")
        
        # Check if requested bots are available
        for bot in args.bot:
            if bot not in available_bots:
                print(f"\n⚠️  Warning: Bot '{bot}' not found in registry")
                print(f"   Available: {', '.join(available_bots)}")
                return 1
    
    # Calculate date range
    try:
        start, end = calculate_date_range(args)
    except ValueError as e:
        _LOG.error(f"Date range error: {e}")
        return 1
    
    # Create backtest configuration
    config = BacktestConfig(
        tickers=[t.upper() for t in args.tickers],
        start=start,
        end=end,
        starting_cash=args.cash,
        sentiment_window_minutes=args.window,
        bar_timeframe=args.timeframe,
        bot_variants=args.bot,
        use_vader=not args.no_vader,
        use_finbert=args.finbert,
        use_keybert=args.keybert,
        record_trace=args.trace,
        max_workers=args.workers or None,
    )
    
    # Run backtest
    _LOG.info(f"Starting backtest: {len(config.tickers)} tickers, {config.bot_variants}")
    _LOG.info(f"Date range: {start.date()} to {end.date()}")
    
    try:
        runner = BacktestRunner(config)
        results = runner.run()
    except Exception as e:
        _LOG.error(f"Backtest failed: {e}", exc_info=True)
        return 1
    
    # Display results
    display_results(results, config, quiet=args.quiet, compare=args.compare)
    
    # Save to file if requested
    if args.output:
        save_results_to_file(results, config, args.output)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())



