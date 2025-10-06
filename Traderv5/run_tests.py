"""Test runner for Traderv5 system."""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest

def run_tests():
    """Run all tests for the Traderv5 system."""
    test_files = [
        "Traderv5/test_cash_ledger.py",
        "Traderv5/test_bucket_scheduler.py", 
        "Traderv5/test_decision_router.py",
        "Traderv5/test_integration.py",
    ]
    
    # Run tests with verbose output
    pytest_args = [
        "-v",
        "--tb=short",
        "--color=yes",
    ] + test_files
    
    exit_code = pytest.main(pytest_args)
    return exit_code

if __name__ == "__main__":
    exit_code = run_tests()
    sys.exit(exit_code)
