#!/bin/bash

# Automated Trading System - Runner Script
# This script provides easy commands to run the trading system

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}$1${NC}"
}

# Function to check if Python is available
check_python() {
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed or not in PATH"
        exit 1
    fi
    
    local python_version=$(python3 --version | cut -d' ' -f2)
    print_status "Using Python $python_version"
}

# Function to check if virtual environment exists
check_venv() {
    if [ ! -d "venv" ]; then
        print_warning "Virtual environment not found. Creating one..."
        python3 -m venv venv
        print_status "Virtual environment created"
    fi
}

# Function to activate virtual environment
activate_venv() {
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
        print_status "Virtual environment activated"
    else
        print_error "Cannot activate virtual environment"
        exit 1
    fi
}

# Function to install dependencies
install_deps() {
    print_status "Installing/updating dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
    print_status "Dependencies installed"
}

# Function to check environment setup
check_env() {
    if [ ! -f ".env" ]; then
        print_warning ".env file not found. Creating sample..."
        python3 main.py --create-env
        print_error "Please edit .env file with your API keys before running the system"
        exit 1
    fi
    
    # Check if API keys are set
    if grep -q "your_alpaca_api_key_here" .env; then
        print_error "Please update .env file with your actual Alpaca API keys"
        exit 1
    fi
    
    print_status "Environment configuration found"
}

# Function to run system check
run_check() {
    print_header "Running System Check..."
    python3 main.py --check
}

# Function to run in dry-run mode
run_dry() {
    print_header "Starting System in Dry-Run Mode..."
    python3 main.py --dry-run
}

# Function to run live trading
run_live() {
    print_header "Starting Live Trading System..."
    print_warning "This will execute real trades in paper trading mode!"
    read -p "Are you sure you want to continue? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python3 main.py
    else
        print_status "Cancelled by user"
    fi
}

# Function to show logs
show_logs() {
    local log_type=${1:-main}
    
    case $log_type in
        "main"|"app")
            tail -f logs/main/app_$(date +%Y-%m-%d).log
            ;;
        "trades"|"trading")
            tail -f logs/trades/trading_$(date +%Y-%m-%d).log
            ;;
        "alerts")
            tail -f logs/alerts/alerts_$(date +%Y-%m-%d).log
            ;;
        "errors")
            tail -f logs/errors/errors_$(date +%Y-%m-%d).log
            ;;
        "performance")
            tail -f logs/performance/performance_$(date +%Y-%m-%d).log
            ;;
        *)
            print_error "Unknown log type: $log_type"
            print_status "Available log types: main, trades, alerts, errors, performance"
            exit 1
            ;;
    esac
}

# Function to clean logs
clean_logs() {
    print_warning "This will delete all log files"
    read -p "Are you sure? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf logs/
        print_status "Log files deleted"
    else
        print_status "Cancelled by user"
    fi
}

# Function to setup the system
setup() {
    print_header "Setting up Automated Trading System..."
    
    check_python
    check_venv
    activate_venv
    install_deps
    check_env
    
    print_status "Setup completed! You can now run the system."
    print_status "Try: ./run_system.sh check"
}

# Function to show help
show_help() {
    echo "Automated Trading System - Runner Script"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  setup          Setup the system (install dependencies, create venv, etc.)"
    echo "  check          Run system health check"
    echo "  dry-run        Start system in dry-run mode (no actual trades)"
    echo "  live           Start live trading system"
    echo "  logs [TYPE]    Show logs (types: main, trades, alerts, errors, performance)"
    echo "  clean-logs     Delete all log files"
    echo "  help           Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 setup       # First time setup"
    echo "  $0 check       # Test system health"
    echo "  $0 dry-run     # Safe testing mode"
    echo "  $0 logs trades # Watch trading logs"
}

# Main script logic
case "${1:-help}" in
    "setup")
        setup
        ;;
    "check")
        check_python
        activate_venv
        check_env
        run_check
        ;;
    "dry-run"|"dry")
        check_python
        activate_venv
        check_env
        run_dry
        ;;
    "live"|"run")
        check_python
        activate_venv
        check_env
        run_live
        ;;
    "logs")
        show_logs "${2:-main}"
        ;;
    "clean-logs")
        clean_logs
        ;;
    "help"|"--help"|"-h")
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac
