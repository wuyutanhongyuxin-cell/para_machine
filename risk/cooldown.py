"""
Cooldown Management module for Paradex Trader.

Implements cooldown periods after:
- Consecutive losses
- Large losses
- High volatility periods
- Manual triggers

Helps prevent emotional trading and overtrading during adverse conditions.
"""

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("paradex_trader.risk.cooldown")


@dataclass
class CooldownEvent:
    """Record of a cooldown trigger."""
    timestamp: float
    reason: str
    duration: float
    expires_at: float
    strategy: Optional[str] = None


class CooldownManager:
    """
    Cooldown period manager.

    Manages multiple types of cooldowns:
    1. Consecutive loss cooldown
    2. Large loss cooldown
    3. Strategy-specific cooldown
    4. Global cooldown
    5. Volatility cooldown
    """

    def __init__(
        self,
        max_consecutive_losses: int = 5,
        cooldown_seconds: int = 300,
        large_loss_threshold: float = 0.02,
        large_loss_cooldown: int = 600,
        volatility_cooldown: int = 300,
        max_cooldown_seconds: int = 3600,
    ):
        """
        Initialize cooldown manager.

        Args:
            max_consecutive_losses: Losses before cooldown triggers.
            cooldown_seconds: Default cooldown duration.
            large_loss_threshold: Loss percentage that triggers large loss cooldown.
            large_loss_cooldown: Cooldown duration for large losses.
            volatility_cooldown: Cooldown duration for high volatility.
            max_cooldown_seconds: Maximum total cooldown duration.
        """
        self.max_consecutive_losses = max_consecutive_losses
        self.cooldown_seconds = cooldown_seconds
        self.large_loss_threshold = large_loss_threshold
        self.large_loss_cooldown = large_loss_cooldown
        self.volatility_cooldown = volatility_cooldown
        self.max_cooldown_seconds = max_cooldown_seconds

        # State tracking
        self._consecutive_losses = 0
        self._last_loss_time: float = 0
        self._global_cooldown_until: float = 0
        self._strategy_cooldowns: Dict[str, float] = {}

        # History
        self._cooldown_history: List[CooldownEvent] = []
        self._trade_results: List[Tuple[float, bool, float]] = []  # (timestamp, is_win, pnl_pct)

        logger.info(
            f"CooldownManager initialized: max_consecutive_losses={max_consecutive_losses}, "
            f"cooldown_seconds={cooldown_seconds}"
        )

    def record_trade_result(
        self,
        is_win: bool,
        pnl_pct: float,
        strategy: Optional[str] = None,
    ) -> Optional[CooldownEvent]:
        """
        Record a trade result and check for cooldown triggers.

        Args:
            is_win: Whether trade was profitable.
            pnl_pct: PnL as percentage.
            strategy: Strategy that made the trade.

        Returns:
            CooldownEvent if cooldown was triggered, None otherwise.
        """
        now = time.time()
        self._trade_results.append((now, is_win, pnl_pct))

        # Keep last 100 trades
        if len(self._trade_results) > 100:
            self._trade_results = self._trade_results[-50:]

        if is_win:
            # Reset consecutive losses on win
            self._consecutive_losses = 0
            return None

        # Record loss
        self._consecutive_losses += 1
        self._last_loss_time = now

        # Check for consecutive loss cooldown
        if self._consecutive_losses >= self.max_consecutive_losses:
            return self._trigger_cooldown(
                reason="consecutive_losses",
                duration=self.cooldown_seconds,
                strategy=strategy,
            )

        # Check for large loss cooldown
        if abs(pnl_pct) >= self.large_loss_threshold:
            return self._trigger_cooldown(
                reason="large_loss",
                duration=self.large_loss_cooldown,
                strategy=strategy,
            )

        return None

    def trigger_volatility_cooldown(self, volatility: float, threshold: float = 0.05) -> Optional[CooldownEvent]:
        """
        Trigger cooldown due to high volatility.

        Args:
            volatility: Current volatility.
            threshold: Volatility threshold.

        Returns:
            CooldownEvent if triggered.
        """
        if volatility >= threshold:
            return self._trigger_cooldown(
                reason="high_volatility",
                duration=self.volatility_cooldown,
            )
        return None

    def trigger_manual_cooldown(self, duration: int, reason: str = "manual") -> CooldownEvent:
        """
        Manually trigger a cooldown.

        Args:
            duration: Cooldown duration in seconds.
            reason: Reason for cooldown.

        Returns:
            CooldownEvent.
        """
        return self._trigger_cooldown(reason=reason, duration=duration)

    def _trigger_cooldown(
        self,
        reason: str,
        duration: float,
        strategy: Optional[str] = None,
    ) -> CooldownEvent:
        """
        Internal method to trigger a cooldown.

        Args:
            reason: Reason for cooldown.
            duration: Duration in seconds.
            strategy: Affected strategy (None for global).

        Returns:
            CooldownEvent.
        """
        now = time.time()
        expires_at = now + min(duration, self.max_cooldown_seconds)

        event = CooldownEvent(
            timestamp=now,
            reason=reason,
            duration=duration,
            expires_at=expires_at,
            strategy=strategy,
        )

        if strategy:
            # Strategy-specific cooldown
            self._strategy_cooldowns[strategy] = expires_at
            logger.warning(
                f"Strategy cooldown triggered: {strategy} - {reason}, "
                f"duration={duration}s"
            )
        else:
            # Global cooldown
            self._global_cooldown_until = expires_at
            logger.warning(
                f"Global cooldown triggered: {reason}, duration={duration}s"
            )

        self._cooldown_history.append(event)

        # Keep last 50 events
        if len(self._cooldown_history) > 50:
            self._cooldown_history = self._cooldown_history[-25:]

        return event

    def is_in_cooldown(self, strategy: Optional[str] = None) -> Tuple[bool, str, float]:
        """
        Check if currently in cooldown.

        Args:
            strategy: Check specific strategy (None for global only).

        Returns:
            Tuple of (is_in_cooldown, reason, remaining_seconds).
        """
        now = time.time()

        # Check global cooldown
        if self._global_cooldown_until > now:
            remaining = self._global_cooldown_until - now
            return True, "global_cooldown", remaining

        # Check strategy-specific cooldown
        if strategy and strategy in self._strategy_cooldowns:
            if self._strategy_cooldowns[strategy] > now:
                remaining = self._strategy_cooldowns[strategy] - now
                return True, f"strategy_cooldown_{strategy}", remaining
            else:
                # Expired, clean up
                del self._strategy_cooldowns[strategy]

        return False, "", 0

    def get_remaining_cooldown(self, strategy: Optional[str] = None) -> float:
        """
        Get remaining cooldown time.

        Args:
            strategy: Check specific strategy.

        Returns:
            Remaining seconds, 0 if not in cooldown.
        """
        in_cooldown, _, remaining = self.is_in_cooldown(strategy)
        return remaining if in_cooldown else 0

    def clear_cooldown(self, strategy: Optional[str] = None) -> None:
        """
        Clear cooldown (manual override).

        Args:
            strategy: Clear specific strategy cooldown (None for global).
        """
        if strategy:
            if strategy in self._strategy_cooldowns:
                del self._strategy_cooldowns[strategy]
                logger.info(f"Strategy cooldown cleared: {strategy}")
        else:
            self._global_cooldown_until = 0
            logger.info("Global cooldown cleared")

    def clear_all_cooldowns(self) -> None:
        """Clear all cooldowns."""
        self._global_cooldown_until = 0
        self._strategy_cooldowns.clear()
        logger.info("All cooldowns cleared")

    def reset_consecutive_losses(self) -> None:
        """Reset consecutive loss counter."""
        self._consecutive_losses = 0
        logger.debug("Consecutive losses reset")

    def get_consecutive_losses(self) -> int:
        """Get current consecutive loss count."""
        return self._consecutive_losses

    def get_recent_win_rate(self, window: int = 20) -> float:
        """
        Get recent win rate.

        Args:
            window: Number of recent trades to consider.

        Returns:
            Win rate as fraction.
        """
        if not self._trade_results:
            return 0.5  # Default

        recent = self._trade_results[-window:]
        wins = sum(1 for _, is_win, _ in recent if is_win)
        return wins / len(recent)

    def should_reduce_risk(self) -> Tuple[bool, float]:
        """
        Check if risk should be reduced based on recent performance.

        Returns:
            Tuple of (should_reduce, multiplier).
        """
        # Reduce if consecutive losses approaching limit
        if self._consecutive_losses >= self.max_consecutive_losses - 2:
            return True, 0.5

        # Reduce if recent win rate is poor
        recent_wr = self.get_recent_win_rate(10)
        if recent_wr < 0.3:
            return True, 0.7

        return False, 1.0

    def get_stats(self) -> dict:
        """Get cooldown statistics."""
        in_global, _, global_remaining = self.is_in_cooldown()

        return {
            "consecutive_losses": self._consecutive_losses,
            "max_consecutive_losses": self.max_consecutive_losses,
            "in_global_cooldown": in_global,
            "global_cooldown_remaining": global_remaining,
            "strategy_cooldowns": {
                k: v - time.time()
                for k, v in self._strategy_cooldowns.items()
                if v > time.time()
            },
            "recent_win_rate": self.get_recent_win_rate(),
            "cooldown_events_count": len(self._cooldown_history),
            "last_loss_time": self._last_loss_time,
        }

    def get_cooldown_history(self, limit: int = 10) -> List[dict]:
        """Get recent cooldown history."""
        events = self._cooldown_history[-limit:]
        return [
            {
                "timestamp": e.timestamp,
                "reason": e.reason,
                "duration": e.duration,
                "strategy": e.strategy,
            }
            for e in events
        ]
