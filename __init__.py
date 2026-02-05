"""
Paradex Trader - Self-Learning Cryptocurrency Trading Bot.

A sophisticated trading bot for Paradex DEX that uses:
- Thompson Sampling for automatic strategy selection
- Online learning for signal quality filtering
- Market regime detection
- Comprehensive risk management

Usage:
    python -m paradex_trader.main [--dry-run] [--debug]
"""

__version__ = "1.0.0"
__author__ = "Paradex Trader Team"

from paradex_trader.config.settings import Settings, load_settings
from paradex_trader.main import TradingEngine

__all__ = [
    "TradingEngine",
    "Settings",
    "load_settings",
    "__version__",
]
