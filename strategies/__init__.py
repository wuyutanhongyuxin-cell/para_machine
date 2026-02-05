"""Trading strategies module."""

from .base import (
    BaseStrategy,
    Signal,
    Direction,
    ExitLevels,
    TradeContext,
    StrategyManager,
)
from .trend_follow import TrendFollowStrategy
from .mean_reversion import MeanReversionStrategy
from .momentum import MomentumStrategy

__all__ = [
    # Base classes
    "BaseStrategy",
    "Signal",
    "Direction",
    "ExitLevels",
    "TradeContext",
    "StrategyManager",
    # Concrete strategies
    "TrendFollowStrategy",
    "MeanReversionStrategy",
    "MomentumStrategy",
]


def create_strategy(name: str, config: dict = None) -> BaseStrategy:
    """
    Factory function to create strategy instances.

    Args:
        name: Strategy name ("trend_follow", "mean_reversion", "momentum").
        config: Strategy configuration.

    Returns:
        Strategy instance.

    Raises:
        ValueError: If strategy name is unknown.
    """
    strategies = {
        "trend_follow": TrendFollowStrategy,
        "mean_reversion": MeanReversionStrategy,
        "momentum": MomentumStrategy,
    }

    if name not in strategies:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(strategies.keys())}")

    return strategies[name](config)


def create_all_strategies(config=None) -> dict:
    """
    Create instances of all available strategies.

    Args:
        config: Configuration object or dictionary.

    Returns:
        Dictionary of strategy name -> strategy instance.
    """
    # Handle Pydantic models by converting to dict or extracting strategy config
    strategy_config = {}
    if config is not None:
        if hasattr(config, 'strategy'):
            # It's a TradingConfig with a strategy attribute
            strategy_config = config.strategy.model_dump() if hasattr(config.strategy, 'model_dump') else {}
        elif hasattr(config, 'model_dump'):
            # It's a Pydantic model
            strategy_config = config.model_dump()
        elif isinstance(config, dict):
            strategy_config = config

    return {
        "trend_follow": TrendFollowStrategy(strategy_config.get("trend_follow")),
        "mean_reversion": MeanReversionStrategy(strategy_config.get("mean_reversion")),
        "momentum": MomentumStrategy(strategy_config.get("momentum")),
    }
