"""Machine learning and adaptive learning module."""

from .thompson_sampling import ThompsonSampling, StrategyArm
from .online_filter import OnlineLearningFilter, PredictionResult
from .feature_engine import FeatureEngine, Candle
from .regime_detector import RegimeDetector, MarketRegime, RegimeState

__all__ = [
    # Thompson Sampling
    "ThompsonSampling",
    "StrategyArm",
    # Online Learning
    "OnlineLearningFilter",
    "PredictionResult",
    # Feature Engineering
    "FeatureEngine",
    "Candle",
    # Regime Detection
    "RegimeDetector",
    "MarketRegime",
    "RegimeState",
]


class LearningManager:
    """
    Unified learning components manager.

    Coordinates Thompson Sampling, Online Learning, and Regime Detection
    for adaptive strategy selection.
    """

    def __init__(
        self,
        strategies: list[str],
        min_trials_per_strategy: int = 20,
        online_learning_enabled: bool = True,
    ):
        """
        Initialize learning manager.

        Args:
            strategies: List of strategy names.
            min_trials_per_strategy: Minimum trials before exploitation.
            online_learning_enabled: Whether to use online learning filter.
        """
        self.thompson = ThompsonSampling(
            strategies=strategies,
            min_trials_per_strategy=min_trials_per_strategy,
        )

        self.online_filter = OnlineLearningFilter() if online_learning_enabled else None
        self.feature_engine = FeatureEngine()
        self.regime_detector = RegimeDetector()

    def update_price(self, price: float, volume: float = 0.0, side: str = None) -> None:
        """Update with new price data."""
        self.feature_engine.update_price(price, volume, side=side)
        self.regime_detector.update(price)

    def update_orderbook(
        self,
        bid: float,
        ask: float,
        imbalance: float,
        bid_depth: float = 0.0,
        ask_depth: float = 0.0,
    ) -> None:
        """Update with orderbook data."""
        self.feature_engine.update_orderbook(bid, ask, imbalance, bid_depth, ask_depth)

    def select_strategy(self) -> str:
        """Select best strategy using Thompson Sampling."""
        return self.thompson.select_strategy()

    def should_trade(self, strategy: str, features: dict = None) -> tuple[bool, float]:
        """
        Check if we should trade using online filter and regime.

        Args:
            strategy: Strategy to evaluate.
            features: Features for prediction.

        Returns:
            Tuple of (should_trade, confidence).
        """
        # Get regime suitability
        regime_info = self.regime_detector.get_regime_for_strategy(strategy)

        if not regime_info["should_trade"]:
            return False, regime_info["suitability"]

        # Check online filter if available
        if self.online_filter and features:
            prediction = self.online_filter.predict(features)
            if prediction.model_ready and not prediction.should_trade:
                return False, prediction.probability

            return True, prediction.probability

        return True, regime_info["suitability"]

    def record_trade_result(
        self,
        strategy: str,
        pnl: float,
        pnl_pct: float,
        is_win: bool,
        features: dict = None,
    ) -> None:
        """
        Record trade result for learning.

        Args:
            strategy: Strategy that was used.
            pnl: Profit/loss amount.
            pnl_pct: Profit/loss percentage.
            is_win: Whether trade was profitable.
            features: Features at trade entry.
        """
        # Update Thompson Sampling
        self.thompson.update(strategy, pnl, pnl_pct, is_win)

        # Update online filter
        if self.online_filter and features:
            self.online_filter.learn(features, is_win)

    def get_features(self) -> dict:
        """Get current features from engine."""
        return self.feature_engine.get_features()

    def get_regime(self) -> RegimeState:
        """Get current market regime."""
        return self.regime_detector.get_current_regime()

    def get_stats(self) -> dict:
        """Get combined learning statistics."""
        stats = {
            "thompson": self.thompson.get_summary(),
            "regime": self.regime_detector.get_stats(),
            "features": self.feature_engine.get_stats(),
        }

        if self.online_filter:
            stats["online_filter"] = self.online_filter.get_stats()

        return stats

    def save_state(self, directory: str) -> None:
        """Save all learning state."""
        from pathlib import Path
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        self.thompson.save_state(str(path / "thompson_state.json"))

        if self.online_filter:
            self.online_filter.save_state(str(path / "online_filter.pkl"))

    def load_state(self, directory: str) -> None:
        """Load all learning state."""
        from pathlib import Path
        path = Path(directory)

        self.thompson.load_state(str(path / "thompson_state.json"))

        if self.online_filter:
            self.online_filter.load_state(str(path / "online_filter.pkl"))

    def reset(self) -> None:
        """Reset all learning components."""
        self.thompson.reset()
        if self.online_filter:
            self.online_filter.reset()
        self.feature_engine.reset()
        self.regime_detector.reset()
