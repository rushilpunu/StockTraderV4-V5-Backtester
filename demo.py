#!/usr/bin/env python3
"""
Demo script to showcase the automated trading system capabilities.
This script demonstrates the system components without requiring API keys.
"""

import sys
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Mock data for demonstration
MOCK_GDELT_DATA = {
    "ticker": "AAPL",
    "company_name": "Apple Inc",
    "search_period": {
        "start": (datetime.utcnow() - timedelta(hours=2)).isoformat(),
        "end": datetime.utcnow().isoformat()
    },
    "articles": [
        {
            "title": "Apple Reports Strong Q4 Earnings, Beats Expectations",
            "url": "https://example.com/apple-earnings",
            "tone": 0.7,
            "seendate": datetime.utcnow().isoformat()
        },
        {
            "title": "Apple Stock Rises on New iPhone Sales Data",
            "url": "https://example.com/apple-iphone-sales", 
            "tone": 0.5,
            "seendate": (datetime.utcnow() - timedelta(minutes=30)).isoformat()
        },
        {
            "title": "Concerns Over Apple Supply Chain Disruptions",
            "url": "https://example.com/apple-supply-chain",
            "tone": -0.3,
            "seendate": (datetime.utcnow() - timedelta(minutes=45)).isoformat()
        }
    ],
    "gkg_data": [
        {
            "themes": "STOCK_MARKET;EARNINGS;TECHNOLOGY;FINANCIAL_PERFORMANCE"
        }
    ],
    "timeline": []
}


async def demo_data_processing():
    """Demonstrate data processing capabilities."""
    print("🔍 Data Processing Demo")
    print("=" * 50)
    
    from src.data_processor import DataProcessor
    
    processor = DataProcessor()
    event_data = processor.process_gdelt_response(MOCK_GDELT_DATA)
    
    print(f"Ticker: {event_data.ticker}")
    print(f"Articles processed: {event_data.event_volume}")
    print(f"Relevance score: {event_data.relevance_score:.3f}")
    print(f"Key themes: {', '.join(event_data.themes[:3])}")
    
    if event_data.sentiment_metrics:
        print(f"Average sentiment: {event_data.sentiment_metrics.average_sentiment:.3f}")
        print(f"Sentiment confidence: {event_data.sentiment_metrics.confidence_score:.3f}")
        print(f"Positive articles: {event_data.sentiment_metrics.positive_count}")
        print(f"Negative articles: {event_data.sentiment_metrics.negative_count}")
    
    print()
    return event_data


async def demo_volatility_analysis(event_data):
    """Demonstrate volatility analysis."""
    print("📊 Volatility Analysis Demo")
    print("=" * 50)
    
    from src.volatility_analyzer import VolatilityAnalyzer
    
    analyzer = VolatilityAnalyzer()
    signal = analyzer.analyze_volatility(event_data)
    
    print(f"Signal type: {signal.signal_type}")
    print(f"Strength: {signal.strength:.3f}")
    print(f"Direction: {signal.direction}")
    print(f"Confidence: {signal.confidence:.3f}")
    print(f"Recommended action: {signal.recommended_action}")
    print(f"Triggers: {', '.join(signal.triggers)}")
    
    print()
    return signal


async def demo_alert_system(event_data, signal):
    """Demonstrate alert generation."""
    print("🚨 Alert System Demo")
    print("=" * 50)
    
    from src.alert_system import AlertSystem
    
    alert_system = AlertSystem()
    alerts = alert_system.process_event_data(event_data, signal)
    
    print(f"Alerts generated: {len(alerts)}")
    
    for i, alert in enumerate(alerts, 1):
        print(f"\nAlert #{i}:")
        print(f"  Type: {alert.alert_type.value}")
        print(f"  Level: {alert.alert_level.value}")
        print(f"  Title: {alert.title}")
        print(f"  Confidence: {alert.confidence:.3f}")
        print(f"  Recommended action: {alert.recommended_action}")
        print(f"  Description: {alert.description[:100]}...")
    
    print()
    return alerts


async def demo_trading_decision(alerts):
    """Demonstrate trading decision logic."""
    print("💰 Trading Decision Demo")
    print("=" * 50)
    
    from src.trading_engine import TradingEngine
    
    trading_engine = TradingEngine()
    
    if not alerts:
        print("No alerts to process")
        return
    
    # Simulate current price
    current_price = 150.00
    
    for alert in alerts:
        decision = await trading_engine.evaluate_trading_decision(
            alert, current_price
        )
        
        if decision:
            print(f"Trading Decision for {alert.ticker}:")
            print(f"  Action: {decision.action.value.upper()}")
            print(f"  Quantity: {decision.quantity} shares")
            print(f"  Order type: {decision.order_type.value}")
            print(f"  Stop loss: ${decision.stop_loss:.2f}" if decision.stop_loss else "  Stop loss: Not set")
            print(f"  Take profit: ${decision.take_profit:.2f}" if decision.take_profit else "  Take profit: Not set")
            print(f"  Risk score: {decision.risk_score:.3f}")
            print(f"  Expected return: {decision.expected_return:.3f}")
            print(f"  Reasoning: {decision.reasoning}")
            print()
        else:
            print(f"No trading decision for {alert.ticker} (risk management or insufficient signal)")
            print()


async def demo_portfolio_summary():
    """Demonstrate portfolio tracking."""
    print("📈 Portfolio Summary Demo")
    print("=" * 50)
    
    from src.trading_engine import TradingEngine
    
    trading_engine = TradingEngine()
    
    # Simulate some positions
    trading_engine.update_position("AAPL", 100, 145.00, "buy")
    trading_engine.update_position("MSFT", 50, 280.00, "buy")
    
    summary = trading_engine.get_portfolio_summary()
    
    print(f"Portfolio Value: ${summary['portfolio_value']:,.2f}")
    print(f"Cash: ${summary['cash']:,.2f}")
    print(f"Positions Value: ${summary['positions_value']:,.2f}")
    print(f"Unrealized P&L: ${summary['unrealized_pnl']:,.2f}")
    print(f"Total Trades: {summary['total_trades']}")
    print(f"Win Rate: {summary['win_rate']:.1f}%")
    
    print("\nPositions:")
    for ticker, pos in summary['positions'].items():
        print(f"  {ticker}: {pos['quantity']} shares @ ${pos['entry_price']:.2f}")
        print(f"    Current: ${pos['current_price']:.2f}")
        print(f"    P&L: ${pos['unrealized_pnl']:.2f} ({pos['unrealized_pnl_percent']:.1f}%)")
    
    print()


async def demo_system_monitoring():
    """Demonstrate system monitoring capabilities."""
    print("🔧 System Monitoring Demo")
    print("=" * 50)
    
    # Simulate system metrics
    system_status = {
        "status": {
            "is_running": True,
            "system_health": "healthy",
            "last_update": datetime.utcnow().isoformat(),
            "consecutive_errors": 0
        },
        "statistics": {
            "total_cycles": 45,
            "successful_cycles": 43,
            "failed_cycles": 2,
            "success_rate": 95.6
        },
        "alerts": {
            "active_count": 2,
            "summary": {
                "total_alerts_generated": 15,
                "alerts_by_level": {
                    "low": 8,
                    "medium": 5,
                    "high": 2,
                    "critical": 0
                }
            }
        },
        "portfolio": {
            "open_positions": 2,
            "daily_pnl": 1250.50
        }
    }
    
    print(f"System Status: {system_status['status']['system_health'].upper()}")
    print(f"Success Rate: {system_status['statistics']['success_rate']:.1f}%")
    print(f"Total Cycles: {system_status['statistics']['total_cycles']}")
    print(f"Active Alerts: {system_status['alerts']['active_count']}")
    print(f"Open Positions: {system_status['portfolio']['open_positions']}")
    print(f"Daily P&L: ${system_status['portfolio']['daily_pnl']:,.2f}")
    
    print()


def print_banner():
    """Print demo banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    Automated Trading System - DEMO                          ║
║                                                                              ║
║  This demo showcases the system's capabilities using mock data              ║
║  No API keys or real trading required                                       ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)


async def main():
    """Main demo function."""
    print_banner()
    
    try:
        # Run through the complete pipeline
        event_data = await demo_data_processing()
        signal = await demo_volatility_analysis(event_data)
        alerts = await demo_alert_system(event_data, signal)
        await demo_trading_decision(alerts)
        await demo_portfolio_summary()
        await demo_system_monitoring()
        
        print("✅ Demo completed successfully!")
        print("\nTo run the actual system:")
        print("1. Set up your API keys: python main.py --create-env")
        print("2. Test the system: python main.py --check")
        print("3. Start dry-run: python main.py --dry-run")
        print("4. Or use the helper script: ./run_system.sh setup")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Demo failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
