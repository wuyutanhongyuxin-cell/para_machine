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

# Lazy imports to avoid circular dependencies
def get_settings():
    """Get Settings class."""
    from paradex_trader.config.settings import Settings
    return Settings

def get_load_settings():
    """Get load_settings function."""
    from paradex_trader.config.settings import load_settings
    return load_settings

def get_trading_engine():
    """Get TradingEngine class."""
    from paradex_trader.main import TradingEngine
    return TradingEngine

__all__ = [
    "get_trading_engine",
    "get_settings",
    "get_load_settings",
    "__version__",
]
