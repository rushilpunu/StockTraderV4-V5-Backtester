#!/usr/bin/env python3
"""Interactive setup script for Alpaca credentials."""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

def setup_alpaca_credentials():
    """Interactive setup for Alpaca credentials."""
    print("🔑 Alpaca API Credentials Setup")
    print("=" * 50)
    print()
    print("To get your Alpaca credentials:")
    print("1. Go to: https://app.alpaca.markets/paper/dashboard/overview")
    print("2. Sign up for a free paper trading account")
    print("3. Go to 'API Keys' section")
    print("4. Generate new API keys")
    print("5. Copy the keys below")
    print()
    
    # Get credentials from user
    api_key = input("Enter your Alpaca API Key: ").strip()
    if not api_key:
        print("❌ API Key is required")
        return False
    
    api_secret = input("Enter your Alpaca API Secret: ").strip()
    if not api_secret:
        print("❌ API Secret is required")
        return False
    
    # Ask for base URL (default to paper trading)
    base_url = input("Enter API Base URL (press Enter for paper trading): ").strip()
    if not base_url:
        base_url = "https://paper-api.alpaca.markets"
    
    print()
    print("📝 Setting up environment variables...")
    
    # Create .env file
    env_file = Path(__file__).parent.parent / ".env"
    
    env_content = f"""# Alpaca Trading API Credentials
ALPACA_API_KEY={api_key}
ALPACA_API_SECRET={api_secret}
ALPACA_API_BASE={base_url}
"""
    
    try:
        with open(env_file, 'w') as f:
            f.write(env_content)
        
        print(f"✅ Credentials saved to: {env_file}")
        
        # Set environment variables for current session
        os.environ['ALPACA_API_KEY'] = api_key
        os.environ['ALPACA_API_SECRET'] = api_secret
        os.environ['ALPACA_API_BASE'] = base_url
        
        print("✅ Environment variables set for current session")
        
        # Test the credentials
        print()
        print("🧪 Testing credentials...")
        
        try:
            from config.credentials import load_alpaca_credentials
            creds = load_alpaca_credentials()
            
            # Try to connect to Alpaca
            from alpaca_trade_api import REST
            client = REST(
                key_id=creds.api_key,
                secret_key=creds.api_secret,
                base_url=creds.base_url
            )
            
            account = client.get_account()
            print(f"✅ Successfully connected to Alpaca!")
            print(f"   Account: {account.account_number}")
            print(f"   Status: {account.status}")
            print(f"   Equity: ${float(account.equity):,.2f}")
            print(f"   Cash: ${float(account.cash):,.2f}")
            
            clock = client.get_clock()
            print(f"   Market: {'OPEN' if clock.is_open else 'CLOSED'}")
            
            return True
            
        except ImportError:
            print("⚠️  alpaca-trade-api not installed. Install with: pip install alpaca-trade-api")
            print("✅ Credentials saved, but connection test skipped")
            return True
            
        except Exception as exc:
            print(f"❌ Connection test failed: {exc}")
            print("✅ Credentials saved, but please verify they are correct")
            return False
            
    except Exception as exc:
        print(f"❌ Failed to save credentials: {exc}")
        return False


def main():
    """Main setup function."""
    if setup_alpaca_credentials():
        print()
        print("🎉 Setup complete! You can now run live trading:")
        print("   cd Traderv5")
        print("   python3 start_live_trading.py")
        return 0
    else:
        print()
        print("❌ Setup failed. Please check your credentials and try again.")
        return 1


if __name__ == "__main__":
    exit(main())
