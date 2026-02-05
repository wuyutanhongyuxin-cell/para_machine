"""
Market Regime Detection for Paradex Trader.

Identifies current market state to help strategies adapt:
- Trending Up/Down
- Ranging/Sideways
- High/Low Volatility
- Breakout/Consolidation

Uses statistical methods and optional HMM for regime classification.
"""

import logging
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("paradex_trader.learning.regime")


class MarketRegime(Enum):
    """Market regime classification."""
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING = "RANGING"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    BREAKOUT_UP = "BREAKOUT_UP"
    BREAKOUT_DOWN = "BREAKOUT_DOWN"
    UNKNOWN = "UNKNOWN"


@dataclass
class RegimeState:
    """Current regime state with confidence."""
    regime: MarketRegime
    confidence: float  # 0 to 1
    secondary_regime: Optional[MarketRegime] = None
    trend_strength: float = 0.0
    volatility_percentile: float = 0.5
    duration_seconds: float = 0.0
    timestamp: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "regime": self.regime.value,
            "confidence": self.confidence,
            "secondary_regime": self.secondary_regime.value if self.secondary_regime else None,
            "trend_strength": self.trend_strength,
            "volatility_percentile": self.volatility_percentile,
            "duration_seconds": self.duration_seconds,
            "timestamp": self.timestamp,
        }


class RegimeDetector:
    """
    Market regime detector using statistical analysis.

    Classifies market into regimes based on:
    - Price trend (direction and strength)
    - Volatility level (percentile)
    - Price range (breakout detection)

    Updates in real-time as new data arrives.
    """

    def __init__(
        self,
        lookback_periods: int = 60,
        trend_threshold: float = 0.3,
        volatility_low_pct: float = 0.25,
        volatility_high_pct: float = 0.75,
        breakout_threshold: float = 0.02,
        min_confidence: float = 0.5,
    ):
        """
        Initialize regime detector.

        Args:
            lookback_periods: Number of periods for analysis.
            trend_threshold: Minimum trend strength to classify as trending.
            volatility_low_pct: Percentile below which volatility is "low".
            volatility_high_pct: Percentile above which volatility is "high".
            breakout_threshold: Price move % to classify as breakout.
            min_confidence: Minimum confidence to report a regime.
        """
        self.lookback_periods = lookback_periods
        self.trend_threshold = trend_threshold
        self.volatility_low_pct = volatility_low_pct
        self.volatility_high_pct = volatility_high_pct
        self.breakout_threshold = breakout_threshold
        self.min_confidence = min_confidence

        # Data storage
        self.prices: Deque[float] = deque(maxlen=lookback_periods * 2)
        self.returns: Deque[float] = deque(maxlen=lookback_periods * 2)
        self.volatilities: Deque[float] = deque(maxlen=lookback_periods)
        self.timestamps: Deque[float] = deque(maxlen=lookback_periods * 2)

        # State tracking
        self._current_regime: Optional[RegimeState] = None
        self._regime_start_time: float = 0
        self._regime_history: List[Tuple[float, MarketRegime]] = []

        logger.info(
            f"RegimeDetector initialized: lookback={lookback_periods}, "
            f"trend_thresh={trend_threshold}"
        )

    def update(self, price: float, timestamp: Optional[float] = None) -> RegimeState:
        """
        Update with new price and detect regime.

        Args:
            price: Current price.
            timestamp: Unix timestamp.

        Returns:
            Current regime state.
        """
        ts = timestamp or time.time()
        self.timestamps.append(ts)

        # Calculate return if we have previous price
        if self.prices:
            ret = (price - self.prices[-1]) / self.prices[-1]
            self.returns.append(ret)

            # Update rolling volatility
            if len(self.returns) >= 10:
                recent_vol = np.std(list(self.returns)[-10:])
                self.volatilities.append(recent_vol)

        self.prices.append(price)

        # Detect current regime
        regime_state = self._detect_regime(ts)

        # Track regime changes
        if self._current_regime is None or regime_state.regime != self._current_regime.regime:
            if self._current_regime is not None:
                self._regime_history.append((ts, self._current_regime.regime))
            self._regime_start_time = ts

        regime_state.duration_seconds = ts - self._regime_start_time
        self._current_regime = regime_state

        return regime_state

    def _detect_regime(self, timestamp: float) -> RegimeState:
        """Detect current market regime."""
        if len(self.prices) < 20:
            return RegimeState(
                regime=MarketRegime.UNKNOWN,
                confidence=0.0,
                timestamp=timestamp,
            )

        prices = list(self.prices)
        returns = list(self.returns) if self.returns else []

        # Calculate trend metrics
        trend_strength = self._calc_trend_strength(prices)
        trend_direction = "up" if trend_strength > 0 else "down"

        # Calculate volatility percentile
        vol_percentile = self._calc_volatility_percentile()
        current_vol = self.volatilities[-1] if self.volatilities else 0

        # Check for breakout
        breakout = self._check_breakout(prices)

        # Determine primary regime
        regime, confidence = self._classify_regime(
            trend_strength, vol_percentile, breakout
        )

        # Determine secondary regime (volatility overlay)
        secondary = None
        if vol_percentile <= self.volatility_low_pct:
            secondary = MarketRegime.LOW_VOLATILITY
        elif vol_percentile >= self.volatility_high_pct:
            secondary = MarketRegime.HIGH_VOLATILITY

        return RegimeState(
            regime=regime,
            confidence=confidence,
            secondary_regime=secondary,
            trend_strength=trend_strength,
            volatility_percentile=vol_percentile,
            timestamp=timestamp,
        )

    def _calc_trend_strength(self, prices: List[float]) -> float:
        """
        Calculate trend strength using multiple methods.

        Returns:
            Trend strength from -1 (strong down) to 1 (strong up).
        """
        if len(prices) < 10:
            return 0.0

        # Method 1: Linear regression slope
        x = np.arange(len(prices))
        slope, _ = np.polyfit(x, prices, 1)
        normalized_slope = slope / np.mean(prices) * len(prices)

        # Method 2: Price position relative to MA
        short_ma = np.mean(prices[-10:])
        long_ma = np.mean(prices[-min(30, len(prices)):])
        ma_signal = (short_ma - long_ma) / long_ma if long_ma > 0 else 0

        # Method 3: Directional movement
        up_moves = sum(1 for i in range(1, len(prices)) if prices[i] > prices[i - 1])
        down_moves = len(prices) - 1 - up_moves
        direction_bias = (up_moves - down_moves) / (len(prices) - 1)

        # Combine methods
        trend = (normalized_slope * 0.4 + ma_signal * 10 * 0.3 + direction_bias * 0.3)

        return max(-1.0, min(1.0, trend))

    def _calc_volatility_percentile(self) -> float:
        """Calculate current volatility percentile."""
        if len(self.volatilities) < 10:
            return 0.5

        vols = list(self.volatilities)
        current = vols[-1]

        # Percentile rank
        below = sum(1 for v in vols if v < current)
        return below / len(vols)

    def _check_breakout(self, prices: List[float]) -> Optional[str]:
        """Check for price breakout."""
        if len(prices) < 30:
            return None

        recent = prices[-5:]
        lookback = prices[-30:-5]

        highest = max(lookback)
        lowest = min(lookback)
        current = recent[-1]

        # Check for upside breakout
        if current > highest * (1 + self.breakout_threshold * 0.5):
            return "up"

        # Check for downside breakout
        if current < lowest * (1 - self.breakout_threshold * 0.5):
            return "down"

        return None

    def _classify_regime(
        self,
        trend_strength: float,
        vol_percentile: float,
        breakout: Optional[str],
    ) -> Tuple[MarketRegime, float]:
        """Classify regime based on metrics."""
        # Breakout takes priority
        if breakout == "up":
            confidence = min(1.0, abs(trend_strength) + 0.3)
            return MarketRegime.BREAKOUT_UP, confidence

        if breakout == "down":
            confidence = min(1.0, abs(trend_strength) + 0.3)
            return MarketRegime.BREAKOUT_DOWN, confidence

        # Trending
        if abs(trend_strength) >= self.trend_threshold:
            confidence = min(1.0, abs(trend_strength))
            if trend_strength > 0:
                return MarketRegime.TRENDING_UP, confidence
            else:
                return MarketRegime.TRENDING_DOWN, confidence

        # Volatility-based for ranging markets
        if vol_percentile >= self.volatility_high_pct:
            return MarketRegime.HIGH_VOLATILITY, vol_percentile

        if vol_percentile <= self.volatility_low_pct:
            return MarketRegime.LOW_VOLATILITY, 1 - vol_percentile

        # Default to ranging
        confidence = 1 - abs(trend_strength) / self.trend_threshold
        return MarketRegime.RANGING, confidence

    def get_current_regime(self) -> RegimeState:
        """Get current regime state."""
        if self._current_regime is None:
            return RegimeState(
                regime=MarketRegime.UNKNOWN,
                confidence=0.0,
                timestamp=time.time(),
            )
        return self._current_regime

    def get_regime_for_strategy(self, strategy: str) -> Dict[str, Any]:
        """
        Get regime suitability for a specific strategy.

        Args:
            strategy: Strategy name.

        Returns:
            Dictionary with suitability score and recommendation.
        """
        regime = self.get_current_regime()

        # Define strategy-regime compatibility
        compatibility = {
            "trend_follow": {
                MarketRegime.TRENDING_UP: 1.0,
                MarketRegime.TRENDING_DOWN: 1.0,
                MarketRegime.BREAKOUT_UP: 0.8,
                MarketRegime.BREAKOUT_DOWN: 0.8,
                MarketRegime.RANGING: 0.2,
                MarketRegime.HIGH_VOLATILITY: 0.5,
                MarketRegime.LOW_VOLATILITY: 0.3,
                MarketRegime.UNKNOWN: 0.5,
            },
            "mean_reversion": {
                MarketRegime.TRENDING_UP: 0.3,
                MarketRegime.TRENDING_DOWN: 0.3,
                MarketRegime.BREAKOUT_UP: 0.1,
                MarketRegime.BREAKOUT_DOWN: 0.1,
                MarketRegime.RANGING: 1.0,
                MarketRegime.HIGH_VOLATILITY: 0.4,
                MarketRegime.LOW_VOLATILITY: 0.8,
                MarketRegime.UNKNOWN: 0.5,
            },
            "momentum": {
                MarketRegime.TRENDING_UP: 0.7,
                MarketRegime.TRENDING_DOWN: 0.7,
                MarketRegime.BREAKOUT_UP: 1.0,
                MarketRegime.BREAKOUT_DOWN: 1.0,
                MarketRegime.RANGING: 0.4,
                MarketRegime.HIGH_VOLATILITY: 0.8,
                MarketRegime.LOW_VOLATILITY: 0.3,
                MarketRegime.UNKNOWN: 0.5,
            },
        }

        strategy_compat = compatibility.get(strategy, {})
        suitability = strategy_compat.get(regime.regime, 0.5)

        # Adjust by confidence
        adjusted_suitability = suitability * regime.confidence + 0.5 * (1 - regime.confidence)

        return {
            "strategy": strategy,
            "regime": regime.regime.value,
            "suitability": adjusted_suitability,
            "should_trade": adjusted_suitability >= 0.5,
            "confidence": regime.confidence,
        }

    def get_strategy_recommendations(self) -> List[Dict[str, Any]]:
        """Get recommendations for all strategies."""
        strategies = ["trend_follow", "mean_reversion", "momentum"]
        recommendations = []

        for strategy in strategies:
            rec = self.get_regime_for_strategy(strategy)
            recommendations.append(rec)

        # Sort by suitability
        recommendations.sort(key=lambda x: x["suitability"], reverse=True)
        return recommendations

    def get_regime_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent regime history."""
        history = self._regime_history[-limit:]
        return [
            {"timestamp": ts, "regime": regime.value}
            for ts, regime in history
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get detector statistics."""
        regime = self.get_current_regime()

        # Calculate regime distribution
        if self._regime_history:
            regime_counts = {}
            for _, r in self._regime_history:
                regime_counts[r.value] = regime_counts.get(r.value, 0) + 1
            total = len(self._regime_history)
            regime_distribution = {k: v / total for k, v in regime_counts.items()}
        else:
            regime_distribution = {}

        return {
            "current_regime": regime.regime.value,
            "confidence": regime.confidence,
            "trend_strength": regime.trend_strength,
            "volatility_percentile": regime.volatility_percentile,
            "duration_seconds": regime.duration_seconds,
            "data_points": len(self.prices),
            "regime_changes": len(self._regime_history),
            "regime_distribution": regime_distribution,
        }

    def reset(self) -> None:
        """Reset detector state."""
        self.prices.clear()
        self.returns.clear()
        self.volatilities.clear()
        self.timestamps.clear()
        self._current_regime = None
        self._regime_start_time = 0
        self._regime_history.clear()

        logger.info("RegimeDetector reset")
