"""
Trend Following Strategy for Paradex Trader.

Based on Donchian Channel breakout with RSI and trend strength filters.
Suitable for medium latency environments (86ms).

Entry Logic:
- LONG: Close > Highest High (20 periods) AND RSI < 75 AND trend_strength > 0.3
- SHORT: Close < Lowest Low (20 periods) AND RSI > 25 AND trend_strength < -0.3

Exit Logic:
- Reverse breakout (10 period channel)
- ATR-based stop loss (2 ATR)
- Trailing take profit

Expected Performance:
- Win Rate: 45-55%
- Risk/Reward: 2.0
- Time Frame: 5-minute candles
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

logger = logging.getLogger("paradex_trader.strategy.trend_follow")


class TrendFollowStrategy(BaseStrategy):
    """
    Trend Following Strategy using Donchian Channel breakouts.

    This strategy identifies strong trends and enters on breakouts,
    using ATR-based stops and trailing take profits.
    """

    # Default parameters
    DEFAULT_CONFIG = {
        # Entry parameters
        "entry_lookback": 20,          # Donchian channel period for entry
        "exit_lookback": 10,           # Donchian channel period for exit
        "min_trend_strength": 0.3,     # Minimum trend strength (-1 to 1)
        "rsi_long_max": 75,            # Max RSI for long entry (avoid overbought)
        "rsi_short_min": 25,           # Min RSI for short entry (avoid oversold)

        # Exit parameters
        "atr_period": 14,              # ATR calculation period
        "stop_loss_atr": 2.0,          # Stop loss in ATR multiples
        "take_profit_atr": 4.0,        # Take profit in ATR multiples
        "trailing_trigger_atr": 2.0,   # Activate trailing at this profit
        "trailing_distance_atr": 1.0,  # Trailing stop distance

        # Filters
        "max_spread_pct": 0.02,        # Maximum spread to enter
        "min_volatility": 0.001,       # Minimum volatility (avoid flat markets)
        "max_volatility": 0.05,        # Maximum volatility (avoid chaos)

        # Time
        "max_hold_minutes": 120,       # Maximum hold time
        "min_signal_interval": 30.0,   # Minimum seconds between signals
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Trend Following Strategy.

        Args:
            config: Strategy configuration (uses defaults if not provided).
        """
        # Merge with defaults
        merged_config = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__("trend_follow", merged_config)

        # Extract frequently used config
        self.entry_lookback = merged_config["entry_lookback"]
        self.exit_lookback = merged_config["exit_lookback"]
        self.min_trend_strength = merged_config["min_trend_strength"]
        self.stop_loss_atr = merged_config["stop_loss_atr"]
        self.take_profit_atr = merged_config["take_profit_atr"]

    async def should_enter(
        self,
        features: Dict[str, float],
        context: TradeContext,
    ) -> Optional[Signal]:
        """
        Check for trend following entry signals.

        Entry conditions for LONG:
        1. Price breaks above 20-period high (Donchian upper)
        2. RSI < 75 (not overbought)
        3. Trend strength > 0.3

        Entry conditions for SHORT:
        1. Price breaks below 20-period low (Donchian lower)
        2. RSI > 25 (not oversold)
        3. Trend strength < -0.3
        """
        # Check common filters
        passed, reason = self.check_entry_filters(features, context)
        if not passed:
            logger.debug(f"[{self.name}] Entry filter failed: {reason}")
            return None

        # Get required features
        donchian_pos = features.get("donchian_position_5m")
        trend_strength = features.get("trend_strength_5m", 0)
        rsi = features.get("rsi_14_5m", 50)
        atr = features.get("atr_14_5m", 0)
        volatility = features.get("volatility_5m", 0)

        # Check if we have required data
        if donchian_pos is None or atr == 0:
            logger.debug(f"[{self.name}] Missing required features")
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

        # Check for LONG breakout
        if donchian_pos >= 0.95:  # Near or above upper channel
            if rsi < self.config["rsi_long_max"] and trend_strength > self.min_trend_strength:
                direction = Direction.LONG
                reason = f"Donchian breakout UP (pos={donchian_pos:.2f}, trend={trend_strength:.2f})"

                # Calculate signal strength
                # Higher trend strength = stronger signal
                strength = min(1.0, 0.5 + abs(trend_strength) + (1 - rsi / 100) * 0.3)

        # Check for SHORT breakout
        elif donchian_pos <= 0.05:  # Near or below lower channel
            if rsi > self.config["rsi_short_min"] and trend_strength < -self.min_trend_strength:
                direction = Direction.SHORT
                reason = f"Donchian breakout DOWN (pos={donchian_pos:.2f}, trend={trend_strength:.2f})"

                # Calculate signal strength
                strength = min(1.0, 0.5 + abs(trend_strength) + (rsi / 100) * 0.3)

        if direction is None:
            return None

        # Create signal
        signal = Signal(
            direction=direction,
            strength=strength,
            strategy=self.name,
            reason=reason,
            features={
                "donchian_position": donchian_pos,
                "trend_strength": trend_strength,
                "rsi": rsi,
                "atr": atr,
                "volatility": volatility,
                "spread_pct": context.spread_pct,
            },
            timestamp=time.time(),
            metadata={
                "entry_lookback": self.entry_lookback,
                "stop_loss_atr": self.stop_loss_atr,
            },
        )

        # Record signal
        self.record_signal(signal)
        self.update_last_signal_time()

        logger.info(
            f"[{self.name}] Signal generated: {direction.value}, "
            f"strength={strength:.2f}, reason={reason}"
        )

        return signal

    def calculate_exit(
        self,
        entry_price: float,
        direction: Direction,
        features: Dict[str, float],
    ) -> ExitLevels:
        """
        Calculate exit levels based on ATR.

        Stop Loss: 2 ATR from entry
        Take Profit: 4 ATR from entry (2:1 R/R)
        Trailing: Activates at 2 ATR profit, trails by 1 ATR
        """
        atr = features.get("atr_14_5m", entry_price * 0.01)  # Default 1% if no ATR

        # Ensure minimum ATR
        min_atr = entry_price * 0.002  # At least 0.2%
        atr = max(atr, min_atr)

        stop_distance = atr * self.stop_loss_atr
        tp_distance = atr * self.take_profit_atr
        trail_trigger = atr * self.config["trailing_trigger_atr"]
        trail_distance = atr * self.config["trailing_distance_atr"]

        if direction == Direction.LONG:
            stop_loss = entry_price - stop_distance
            take_profit = entry_price + tp_distance
            trailing_trigger = entry_price + trail_trigger
        else:  # SHORT
            stop_loss = entry_price + stop_distance
            take_profit = entry_price - tp_distance
            trailing_trigger = entry_price - trail_trigger

        exit_levels = ExitLevels(
            take_profit=take_profit,
            stop_loss=stop_loss,
            trailing_trigger=trailing_trigger,
            trailing_distance=trail_distance,
            time_stop_seconds=self.config["max_hold_minutes"] * 60,
        )

        logger.debug(
            f"[{self.name}] Exit levels: SL={stop_loss:.2f}, TP={take_profit:.2f}, "
            f"Trail trigger={trailing_trigger:.2f}"
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
        Check for early exit conditions.

        Additional exit conditions for trend following:
        1. Trend reversal (trend strength flips sign)
        2. Exit channel breakout (10-period)
        3. Time stop
        """
        # Time stop
        if exit_levels.time_stop_seconds and hold_time >= exit_levels.time_stop_seconds:
            return True, "time_stop"

        # Get current trend strength
        trend_strength = features.get("trend_strength_5m", 0)
        donchian_pos = features.get("donchian_position_5m", 0.5)

        # Trend reversal check
        if direction == Direction.LONG:
            # Exit if trend turns strongly bearish
            if trend_strength < -0.5:
                return True, "trend_reversal"

            # Exit channel breakout (price drops to lower channel)
            if donchian_pos < 0.1:
                return True, "exit_channel_breakout"

        else:  # SHORT
            # Exit if trend turns strongly bullish
            if trend_strength > 0.5:
                return True, "trend_reversal"

            # Exit channel breakout (price rises to upper channel)
            if donchian_pos > 0.9:
                return True, "exit_channel_breakout"

        return False, ""

    def get_position_size_multiplier(
        self,
        signal: Signal,
        features: Dict[str, float],
    ) -> float:
        """
        Adjust position size based on signal quality.

        Higher multiplier for:
        - Strong trend signals
        - Moderate volatility
        - Clear RSI confirmation
        """
        base_mult = signal.strength

        # Adjust for trend strength
        trend = abs(features.get("trend_strength_5m", 0))
        if trend > 0.6:
            base_mult *= 1.2
        elif trend < 0.3:
            base_mult *= 0.8

        # Adjust for volatility (prefer moderate)
        vol = features.get("volatility_5m", 0.02)
        if 0.01 < vol < 0.03:
            base_mult *= 1.1  # Sweet spot
        elif vol > 0.04:
            base_mult *= 0.8  # Reduce in high volatility

        return max(0.5, min(2.0, base_mult))
