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


def create_all_strategies(config: dict = None) -> list[BaseStrategy]:
    """
    Create instances of all available strategies.

    Args:
        config: Configuration dictionary with strategy-specific configs.

    Returns:
        List of strategy instances.
    """
    config = config or {}

    return [
        TrendFollowStrategy(config.get("trend_follow")),
        MeanReversionStrategy(config.get("mean_reversion")),
        MomentumStrategy(config.get("momentum")),
    ]
