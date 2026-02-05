"""
Mean Reversion Strategy for Paradex Trader.

Based on statistical extremes (Z-score) with RSI confirmation.
Trades price reversion to mean after extreme deviations.

Entry Logic:
- LONG: Z-score < -2.0 AND RSI < 25 AND spread < 0.015%
- SHORT: Z-score > 2.0 AND RSI > 75 AND spread < 0.015%

Exit Logic:
- Z-score returns to ±0.5
- Fixed stop loss (2%)
- Time stop (30 minutes)

Expected Performance:
- Win Rate: 55-65%
- Risk/Reward: 0.8-1.0
- Time Frame: 30 second to 5 minute
"""

import logging
import time
from typing import Any, Dict, Optional

from strategies.base import (
    BaseStrategy,
    Direction,
    ExitLevels,
    Signal,
    TradeContext,
)

logger = logging.getLogger("paradex_trader.strategy.mean_reversion")


class MeanReversionStrategy(BaseStrategy):
    """
    Mean Reversion Strategy using Z-score extremes.

    This strategy identifies price extremes and trades the reversion
    back to the mean, using RSI for confirmation and strict filters
    to avoid trending markets.
    """

    # Default parameters
    DEFAULT_CONFIG = {
        # Entry parameters
        "entry_z": 2.0,                # Z-score threshold for entry
        "exit_z": 0.5,                 # Z-score threshold for exit
        "rsi_oversold": 25,            # RSI threshold for long entry
        "rsi_overbought": 75,          # RSI threshold for short entry

        # Exit parameters
        "stop_loss_pct": 0.02,         # Fixed stop loss (2%)
        "take_profit_pct": 0.015,      # Take profit (1.5%)
        "max_hold_minutes": 30,        # Maximum hold time

        # Filters
        "max_spread_pct": 0.015,       # Maximum spread (1.5%)
        "max_trend_strength": 0.3,     # Avoid trending markets
        "min_volatility": 0.005,       # Minimum volatility
        "max_volatility": 0.03,        # Maximum volatility

        # Signal requirements
        "min_signal_strength": 0.5,
        "min_signal_interval": 10.0,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Mean Reversion Strategy.

        Args:
            config: Strategy configuration (uses defaults if not provided).
        """
        merged_config = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__("mean_reversion", merged_config)

        self.entry_z = merged_config["entry_z"]
        self.exit_z = merged_config["exit_z"]
        self.rsi_oversold = merged_config["rsi_oversold"]
        self.rsi_overbought = merged_config["rsi_overbought"]

    async def should_enter(
        self,
        features: Dict[str, float],
        context: TradeContext,
    ) -> Optional[Signal]:
        """
        Check for mean reversion entry signals.

        Entry conditions for LONG (buy the dip):
        1. Z-score < -2.0 (price significantly below mean)
        2. RSI < 25 (extremely oversold)
        3. Spread < 0.015% (good liquidity)
        4. Not in strong trend

        Entry conditions for SHORT (sell the rip):
        1. Z-score > 2.0 (price significantly above mean)
        2. RSI > 75 (extremely overbought)
        3. Spread < 0.015%
        4. Not in strong trend
        """
        # Check common filters
        passed, reason = self.check_entry_filters(features, context)
        if not passed:
            logger.debug(f"[{self.name}] Entry filter failed: {reason}")
            return None

        # Strict spread filter for mean reversion
        if context.spread_pct > self.config["max_spread_pct"]:
            logger.debug(f"[{self.name}] Spread too high for MR: {context.spread_pct:.4f}")
            return None

        # Get required features
        bollinger_pos = features.get("bollinger_position_5m")  # -1 to 1 scale
        rsi = features.get("rsi_14_1m", 50)
        trend_strength = features.get("trend_strength_5m", 0)
        volatility = features.get("volatility_5m", 0)

        # Check required features
        if bollinger_pos is None:
            logger.debug(f"[{self.name}] Missing bollinger position")
            return None

        # Calculate approximate Z-score from Bollinger position
        # bollinger_position is already normalized: -1 = lower band, 1 = upper band
        z_score = bollinger_pos * 2  # Scale to approximate Z-score

        # Trend filter - avoid trading MR in strong trends
        if abs(trend_strength) > self.config["max_trend_strength"]:
            logger.debug(
                f"[{self.name}] Trend too strong for MR: {trend_strength:.2f}"
            )
            return None

        # Volatility filters
        if volatility < self.config["min_volatility"]:
            logger.debug(f"[{self.name}] Volatility too low: {volatility}")
            return None

        if volatility > self.config["max_volatility"]:
            logger.debug(f"[{self.name}] Volatility too high: {volatility}")
            return None

        direction = None
        reason = ""
        strength = 0.0

        # Check for LONG (oversold)
        if z_score < -self.entry_z and rsi < self.rsi_oversold:
            direction = Direction.LONG
            reason = f"Mean reversion LONG (z={z_score:.2f}, RSI={rsi:.1f})"

            # Signal strength based on extremity
            z_extreme = min(abs(z_score) / 3.0, 1.0)  # Normalize Z to 0-1
            rsi_extreme = (self.rsi_oversold - rsi) / self.rsi_oversold
            strength = 0.4 + (z_extreme * 0.3) + (rsi_extreme * 0.3)

        # Check for SHORT (overbought)
        elif z_score > self.entry_z and rsi > self.rsi_overbought:
            direction = Direction.SHORT
            reason = f"Mean reversion SHORT (z={z_score:.2f}, RSI={rsi:.1f})"

            # Signal strength
            z_extreme = min(abs(z_score) / 3.0, 1.0)
            rsi_extreme = (rsi - self.rsi_overbought) / (100 - self.rsi_overbought)
            strength = 0.4 + (z_extreme * 0.3) + (rsi_extreme * 0.3)

        if direction is None:
            return None

        # Minimum strength filter
        if strength < self.config["min_signal_strength"]:
            logger.debug(f"[{self.name}] Signal too weak: {strength:.2f}")
            return None

        # Create signal
        signal = Signal(
            direction=direction,
            strength=min(1.0, strength),
            strategy=self.name,
            reason=reason,
            features={
                "z_score": z_score,
                "bollinger_position": bollinger_pos,
                "rsi": rsi,
                "trend_strength": trend_strength,
                "volatility": volatility,
                "spread_pct": context.spread_pct,
            },
            timestamp=time.time(),
            metadata={
                "entry_z": self.entry_z,
                "exit_z": self.exit_z,
            },
        )

        self.record_signal(signal)
        self.update_last_signal_time()

        logger.info(
            f"[{self.name}] Signal generated: {direction.value}, "
            f"strength={strength:.2f}, z={z_score:.2f}, RSI={rsi:.1f}"
        )

        return signal

    def calculate_exit(
        self,
        entry_price: float,
        direction: Direction,
        features: Dict[str, float],
    ) -> ExitLevels:
        """
        Calculate exit levels for mean reversion.

        Uses fixed percentage stops since we're expecting a quick reversion.
        """
        stop_pct = self.config["stop_loss_pct"]
        tp_pct = self.config["take_profit_pct"]

        if direction == Direction.LONG:
            stop_loss = entry_price * (1 - stop_pct)
            take_profit = entry_price * (1 + tp_pct)
        else:  # SHORT
            stop_loss = entry_price * (1 + stop_pct)
            take_profit = entry_price * (1 - tp_pct)

        exit_levels = ExitLevels(
            take_profit=take_profit,
            stop_loss=stop_loss,
            trailing_trigger=None,  # No trailing for MR
            trailing_distance=None,
            time_stop_seconds=self.config["max_hold_minutes"] * 60,
        )

        logger.debug(
            f"[{self.name}] Exit levels: SL={stop_loss:.2f} ({stop_pct:.1%}), "
            f"TP={take_profit:.2f} ({tp_pct:.1%})"
        )

        return exit_levels

    def should_exit_early(
        self,
        current_price: float,
        entry_price: float,
        direction: Direction,
        hold_time: float,
        features: Dict[str, float],
        exit_levels: ExitLevels,
    ) -> tuple[bool, str]:
        """
        Check for early exit on mean reversion.

        Exit when:
        1. Z-score returns to ±0.5 (mean reversion complete)
        2. RSI returns to neutral zone
        3. Time stop reached
        """
        # Time stop
        if exit_levels.time_stop_seconds and hold_time >= exit_levels.time_stop_seconds:
            return True, "time_stop"

        # Get current Z-score approximation
        bollinger_pos = features.get("bollinger_position_5m", 0)
        z_score = bollinger_pos * 2
        rsi = features.get("rsi_14_1m", 50)

        # Check if mean reversion is complete
        if direction == Direction.LONG:
            # Exit when price returns to mean
            if z_score > -self.exit_z:
                return True, "mean_reversion_complete"

            # Exit if RSI returns to neutral
            if rsi > 50:
                return True, "rsi_neutral"

        else:  # SHORT
            if z_score < self.exit_z:
                return True, "mean_reversion_complete"

            if rsi < 50:
                return True, "rsi_neutral"

        # Check for adverse trend development
        trend_strength = features.get("trend_strength_5m", 0)

        if direction == Direction.LONG and trend_strength < -0.6:
            return True, "adverse_trend"

        if direction == Direction.SHORT and trend_strength > 0.6:
            return True, "adverse_trend"

        return False, ""

    def get_position_size_multiplier(
        self,
        signal: Signal,
        features: Dict[str, float],
    ) -> float:
        """
        Adjust position size for mean reversion.

        Conservative sizing for MR:
        - Reduce size in higher volatility
        - Increase slightly for extreme Z-scores
        """
        base_mult = 0.8  # Start conservative for MR

        # Adjust for Z-score extremity
        z_score = abs(signal.features.get("z_score", 2.0))
        if z_score > 2.5:
            base_mult *= 1.2  # More extreme = more confident
        elif z_score < 1.5:
            base_mult *= 0.7  # Less extreme = less confident

        # Reduce in higher volatility
        vol = features.get("volatility_5m", 0.02)
        if vol > 0.025:
            base_mult *= 0.8

        # Factor in signal strength
        base_mult *= (0.7 + signal.strength * 0.3)

        return max(0.5, min(1.5, base_mult))  # Cap at 1.5x for MR
