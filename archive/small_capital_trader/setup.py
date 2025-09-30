#!/usr/bin/env python3
"""
Setup script for Small Capital Trading System
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = []
with open("requirements.txt", "r") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="small-capital-trader",
    version="1.0.0",
    author="StockTrader Monorepo Team",
    author_email="your-email@example.com",
    description="A specialized variant of the StockTrader (V4/V5) monorepo optimized for small capital ($1,000) trading",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rushilpunu/StockTraderV4-V5-Backtester",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
        ],
        "options": [
            "mibian>=0.1.5",
            "quantlib>=1.25.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "small-capital-trader=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
)
