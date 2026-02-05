"""
Momentum Breakout Strategy for Paradex Trader.

Trades short-term momentum continuation with volume confirmation.
Best suited for volatile market conditions with clear directional moves.

Entry Logic:
- 1-minute price change > 0.1%
- Volume > 2x average
- RSI in 40-60 range (not extreme)
- Orderbook imbalance confirms direction > 0.4

Exit Logic:
- Fixed take profit: 0.15%
- Fixed stop loss: 0.08%
- Time stop: 3 minutes

Expected Performance:
- Win Rate: 50-55%
- Risk/Reward: 1.8
- Time Frame: 1 minute
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

logger = logging.getLogger("paradex_trader.strategy.momentum")


class MomentumStrategy(BaseStrategy):
    """
    Momentum Breakout Strategy.

    This strategy captures short-term momentum continuation by:
    1. Identifying strong price moves
    2. Confirming with volume spike
    3. Using orderbook imbalance for direction
    """

    # Default parameters
    DEFAULT_CONFIG = {
        # Entry parameters
        "min_momentum_pct": 0.1,       # Minimum 1-min price change (0.1%)
        "volume_ratio": 2.0,           # Volume must be 2x average
        "min_imbalance": 0.4,          # Minimum orderbook imbalance
        "rsi_min": 40,                 # RSI lower bound
        "rsi_max": 60,                 # RSI upper bound (neutral zone)

        # Exit parameters
        "take_profit_pct": 0.0015,     # 0.15% take profit
        "stop_loss_pct": 0.0008,       # 0.08% stop loss
        "max_hold_minutes": 3,         # Maximum hold time

        # Filters
        "max_spread_pct": 0.01,        # Tight spread required
        "min_volatility": 0.01,        # Need some volatility
        "max_volatility": 0.04,        # But not too much

        # Signal
        "min_signal_strength": 0.5,
        "min_signal_interval": 5.0,    # Can signal frequently
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Momentum Strategy.

        Args:
            config: Strategy configuration.
        """
        merged_config = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__("momentum", merged_config)

        self.min_momentum = merged_config["min_momentum_pct"]
        self.volume_ratio = merged_config["volume_ratio"]
        self.min_imbalance = merged_config["min_imbalance"]

    async def should_enter(
        self,
        features: Dict[str, float],
        context: TradeContext,
    ) -> Optional[Signal]:
        """
        Check for momentum entry signals.

        Entry conditions for LONG:
        1. 1-minute price change > 0.1%
        2. Volume > 2x average
        3. RSI in 40-60 (not overbought)
        4. Orderbook imbalance > 0.4 (buy pressure)

        Entry conditions for SHORT:
        1. 1-minute price change < -0.1%
        2. Volume > 2x average
        3. RSI in 40-60 (not oversold)
        4. Orderbook imbalance < -0.4 (sell pressure)
        """
        # Check common filters
        passed, reason = self.check_entry_filters(features, context)
        if not passed:
            logger.debug(f"[{self.name}] Entry filter failed: {reason}")
            return None

        # Strict spread filter for momentum
        if context.spread_pct > self.config["max_spread_pct"]:
            logger.debug(f"[{self.name}] Spread too high: {context.spread_pct:.4f}")
            return None

        # Get required features
        price_change_1m = features.get("price_change_1m", 0)
        momentum_1m = features.get("momentum_1m", 0)
        volume_ratio = features.get("volume_ratio", 1.0)
        rsi = features.get("rsi_14_1m", 50)
        imbalance = features.get("imbalance", 0)
        volatility = features.get("volatility_1m", 0)

        # Volatility filters
        if volatility < self.config["min_volatility"]:
            logger.debug(f"[{self.name}] Volatility too low: {volatility}")
            return None

        if volatility > self.config["max_volatility"]:
            logger.debug(f"[{self.name}] Volatility too high: {volatility}")
            return None

        # RSI filter - must be in neutral zone
        if not (self.config["rsi_min"] <= rsi <= self.config["rsi_max"]):
            logger.debug(f"[{self.name}] RSI not in neutral zone: {rsi}")
            return None

        # Volume confirmation
        if volume_ratio < self.volume_ratio:
            logger.debug(f"[{self.name}] Volume too low: {volume_ratio:.2f}x")
            return None

        direction = None
        reason = ""
        strength = 0.0

        # Check for LONG momentum
        if price_change_1m > self.min_momentum:
            if imbalance > self.min_imbalance or context.imbalance > self.min_imbalance:
                direction = Direction.LONG
                reason = (
                    f"Momentum UP (change={price_change_1m:.2f}%, "
                    f"vol={volume_ratio:.1f}x, imb={imbalance:.2f})"
                )

                # Signal strength from momentum and volume
                mom_score = min(abs(price_change_1m) / 0.3, 1.0)
                vol_score = min(volume_ratio / 4.0, 1.0)
                imb_score = min(abs(imbalance), 1.0)

                strength = 0.3 + (mom_score * 0.3) + (vol_score * 0.2) + (imb_score * 0.2)

        # Check for SHORT momentum
        elif price_change_1m < -self.min_momentum:
            effective_imbalance = imbalance if imbalance != 0 else context.imbalance
            if effective_imbalance < -self.min_imbalance:
                direction = Direction.SHORT
                reason = (
                    f"Momentum DOWN (change={price_change_1m:.2f}%, "
                    f"vol={volume_ratio:.1f}x, imb={effective_imbalance:.2f})"
                )

                mom_score = min(abs(price_change_1m) / 0.3, 1.0)
                vol_score = min(volume_ratio / 4.0, 1.0)
                imb_score = min(abs(effective_imbalance), 1.0)

                strength = 0.3 + (mom_score * 0.3) + (vol_score * 0.2) + (imb_score * 0.2)

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
                "price_change_1m": price_change_1m,
                "momentum_1m": momentum_1m,
                "volume_ratio": volume_ratio,
                "rsi": rsi,
                "imbalance": imbalance,
                "volatility": volatility,
                "spread_pct": context.spread_pct,
            },
            timestamp=time.time(),
        )

        self.record_signal(signal)
        self.update_last_signal_time()

        logger.info(
            f"[{self.name}] Signal generated: {direction.value}, "
            f"strength={strength:.2f}, momentum={price_change_1m:.2f}%"
        )

        return signal

    def calculate_exit(
        self,
        entry_price: float,
        direction: Direction,
        features: Dict[str, float],
    ) -> ExitLevels:
        """
        Calculate exit levels for momentum trade.

        Uses tight fixed percentages for quick scalp trades.
        """
        tp_pct = self.config["take_profit_pct"]
        sl_pct = self.config["stop_loss_pct"]

        if direction == Direction.LONG:
            take_profit = entry_price * (1 + tp_pct)
            stop_loss = entry_price * (1 - sl_pct)
        else:  # SHORT
            take_profit = entry_price * (1 - tp_pct)
            stop_loss = entry_price * (1 + sl_pct)

        exit_levels = ExitLevels(
            take_profit=take_profit,
            stop_loss=stop_loss,
            trailing_trigger=None,  # No trailing for quick scalps
            trailing_distance=None,
            time_stop_seconds=self.config["max_hold_minutes"] * 60,
        )

        logger.debug(
            f"[{self.name}] Exit levels: SL={stop_loss:.2f}, TP={take_profit:.2f}"
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
        Check for early exit on momentum trade.

        Exit when:
        1. Momentum reverses
        2. Imbalance flips
        3. Time stop
        """
        # Time stop
        if exit_levels.time_stop_seconds and hold_time >= exit_levels.time_stop_seconds:
            return True, "time_stop"

        # Get current momentum
        momentum = features.get("momentum_1m", 0)
        imbalance = features.get("imbalance", 0)

        # Quick exit on momentum reversal
        if direction == Direction.LONG:
            if momentum < -0.2:  # Momentum turned negative
                return True, "momentum_reversal"

            if imbalance < -0.3:  # Orderbook flipped
                return True, "imbalance_flip"

        else:  # SHORT
            if momentum > 0.2:
                return True, "momentum_reversal"

            if imbalance > 0.3:
                return True, "imbalance_flip"

        # Exit if stuck (no movement after 1 minute)
        if hold_time > 60:
            pnl_pct = (current_price - entry_price) / entry_price
            if direction == Direction.SHORT:
                pnl_pct = -pnl_pct

            if abs(pnl_pct) < 0.0003:  # Less than 0.03% move
                return True, "stuck_position"

        return False, ""

    def get_position_size_multiplier(
        self,
        signal: Signal,
        features: Dict[str, float],
    ) -> float:
        """
        Adjust position size for momentum trade.

        Momentum trades use moderate sizing:
        - Higher size for strong signals
        - Reduce in higher volatility
        """
        base_mult = signal.strength

        # Volume confirmation boost
        vol_ratio = signal.features.get("volume_ratio", 1.0)
        if vol_ratio > 3.0:
            base_mult *= 1.2

        # Reduce in high volatility
        volatility = features.get("volatility_1m", 0.02)
        if volatility > 0.03:
            base_mult *= 0.8

        # Imbalance boost
        imb = abs(signal.features.get("imbalance", 0))
        if imb > 0.6:
            base_mult *= 1.1

        return max(0.5, min(1.5, base_mult))
