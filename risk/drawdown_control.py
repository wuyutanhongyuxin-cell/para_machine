"""
Drawdown Control module for Paradex Trader.

Implements multi-level drawdown protection:
- Per-trade loss limits
- Daily loss limits
- Total drawdown limits

Automatically adjusts trading behavior based on drawdown levels.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

from core.exceptions import RiskLimitExceeded

logger = logging.getLogger("paradex_trader.risk.drawdown")


@dataclass
class DrawdownLevel:
    """Drawdown threshold and action."""
    threshold: float  # Drawdown percentage (e.g., 0.05 = 5%)
    action: str  # "reduce", "pause", "stop"
    description: str


@dataclass
class DailyPnL:
    """Daily PnL tracking."""
    date: str
    pnl: float = 0.0
    trades: int = 0
    wins: int = 0
    peak_pnl: float = 0.0
    max_drawdown: float = 0.0


class DrawdownController:
    """
    Drawdown controller with multi-level protection.

    Levels:
    1. Per-trade: Maximum loss per single trade
    2. Daily: Maximum loss per day
    3. Total: Maximum total drawdown from peak

    Actions:
    - reduce: Reduce position sizes by 50%
    - pause: Pause trading temporarily
    - stop: Stop trading until manual reset
    """

    DEFAULT_LEVELS = [
        DrawdownLevel(0.03, "reduce", "3% drawdown - reduce position sizes"),
        DrawdownLevel(0.05, "pause", "5% drawdown - pause trading"),
        DrawdownLevel(0.10, "stop", "10% drawdown - stop trading"),
    ]

    def __init__(
        self,
        initial_balance: float,
        max_daily_loss_pct: float = 0.05,
        max_total_drawdown_pct: float = 0.15,
        max_loss_per_trade_pct: float = 0.03,
        levels: Optional[List[DrawdownLevel]] = None,
    ):
        """
        Initialize drawdown controller.

        Args:
            initial_balance: Starting account balance.
            max_daily_loss_pct: Maximum daily loss as percentage.
            max_total_drawdown_pct: Maximum total drawdown.
            max_loss_per_trade_pct: Maximum loss per trade.
            levels: Custom drawdown levels (uses defaults if None).
        """
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.peak_balance = initial_balance

        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_total_drawdown_pct = max_total_drawdown_pct
        self.max_loss_per_trade_pct = max_loss_per_trade_pct

        self.levels = levels or self.DEFAULT_LEVELS

        # Daily tracking
        self._daily_pnl: dict[str, DailyPnL] = {}
        self._today_key = self._get_date_key()

        # State
        self._is_paused = False
        self._is_stopped = False
        self._pause_until: Optional[float] = None
        self._position_multiplier = 1.0

        # History
        self._pnl_history: List[Tuple[float, float]] = []  # (timestamp, pnl)
        self._balance_history: List[Tuple[float, float]] = []  # (timestamp, balance)

        logger.info(
            f"DrawdownController initialized: initial_balance={initial_balance}, "
            f"max_daily={max_daily_loss_pct:.1%}, max_total={max_total_drawdown_pct:.1%}"
        )

    def _get_date_key(self) -> str:
        """Get current date key."""
        return datetime.utcnow().strftime("%Y-%m-%d")

    def _get_today(self) -> DailyPnL:
        """Get or create today's PnL record."""
        key = self._get_date_key()
        if key not in self._daily_pnl:
            self._daily_pnl[key] = DailyPnL(date=key)
        return self._daily_pnl[key]

    def update_balance(self, new_balance: float) -> None:
        """
        Update current balance and check drawdown levels.

        Args:
            new_balance: New account balance.
        """
        old_balance = self.current_balance
        self.current_balance = new_balance

        # Update peak
        if new_balance > self.peak_balance:
            self.peak_balance = new_balance

        # Record history
        now = time.time()
        self._balance_history.append((now, new_balance))

        # Keep last 1000 entries
        if len(self._balance_history) > 1000:
            self._balance_history = self._balance_history[-500:]

        # Check drawdown levels
        self._check_drawdown_levels()

    def record_trade(self, pnl: float, is_win: bool) -> None:
        """
        Record a completed trade.

        Args:
            pnl: Trade profit/loss.
            is_win: Whether trade was profitable.
        """
        now = time.time()
        self._pnl_history.append((now, pnl))

        # Update daily stats
        today = self._get_today()
        today.pnl += pnl
        today.trades += 1
        if is_win:
            today.wins += 1

        # Track daily peak and drawdown
        if today.pnl > today.peak_pnl:
            today.peak_pnl = today.pnl

        daily_dd = (today.peak_pnl - today.pnl) if today.peak_pnl > 0 else abs(today.pnl)
        if daily_dd > today.max_drawdown:
            today.max_drawdown = daily_dd

        # Update balance
        self.current_balance += pnl
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance

        self._balance_history.append((now, self.current_balance))

        # Check limits
        self._check_drawdown_levels()
        self._check_daily_limit()

        logger.debug(
            f"Trade recorded: pnl={pnl:+.4f}, daily_pnl={today.pnl:+.4f}, "
            f"balance={self.current_balance:.2f}"
        )

    def _check_drawdown_levels(self) -> None:
        """Check and apply drawdown level actions."""
        current_drawdown = self.get_current_drawdown()

        for level in sorted(self.levels, key=lambda x: x.threshold, reverse=True):
            if current_drawdown >= level.threshold:
                self._apply_action(level)
                break
        else:
            # No level triggered, reset state
            self._position_multiplier = 1.0
            if self._is_paused and self._pause_until and time.time() >= self._pause_until:
                self._is_paused = False
                self._pause_until = None

    def _apply_action(self, level: DrawdownLevel) -> None:
        """Apply drawdown level action."""
        if level.action == "reduce":
            self._position_multiplier = 0.5
            logger.warning(f"Drawdown level triggered: {level.description}")

        elif level.action == "pause":
            if not self._is_paused:
                self._is_paused = True
                self._pause_until = time.time() + 3600  # 1 hour pause
                logger.warning(
                    f"Trading PAUSED: {level.description}. "
                    f"Resume in 1 hour or manual reset."
                )

        elif level.action == "stop":
            self._is_stopped = True
            logger.critical(
                f"Trading STOPPED: {level.description}. "
                f"Manual intervention required."
            )

    def _check_daily_limit(self) -> None:
        """Check daily loss limit."""
        today = self._get_today()
        daily_loss_pct = abs(today.pnl) / self.initial_balance if today.pnl < 0 else 0

        if daily_loss_pct >= self.max_daily_loss_pct:
            if not self._is_paused:
                self._is_paused = True
                # Pause until next day
                tomorrow = datetime.utcnow().replace(
                    hour=0, minute=0, second=0, microsecond=0
                ) + timedelta(days=1)
                self._pause_until = tomorrow.timestamp()

                logger.warning(
                    f"Daily loss limit reached: {daily_loss_pct:.1%}. "
                    f"Trading paused until tomorrow."
                )

    def can_trade(self) -> Tuple[bool, str]:
        """
        Check if trading is allowed.

        Returns:
            Tuple of (can_trade, reason).
        """
        if self._is_stopped:
            return False, "stopped_drawdown_limit"

        if self._is_paused:
            if self._pause_until and time.time() >= self._pause_until:
                self._is_paused = False
                self._pause_until = None
                return True, "ok"
            return False, "paused_drawdown"

        return True, "ok"

    def validate_trade(self, potential_loss: float) -> bool:
        """
        Validate if a trade's potential loss is acceptable.

        Args:
            potential_loss: Maximum potential loss for the trade.

        Returns:
            True if trade is acceptable.

        Raises:
            RiskLimitExceeded: If trade would exceed limits.
        """
        # Check if trading is allowed
        can_trade, reason = self.can_trade()
        if not can_trade:
            raise RiskLimitExceeded(
                f"Trading not allowed: {reason}",
                limit_type="trading_status",
                current_value=0,
                limit_value=0,
            )

        # Check per-trade limit
        max_trade_loss = self.current_balance * self.max_loss_per_trade_pct
        if potential_loss > max_trade_loss:
            raise RiskLimitExceeded(
                f"Trade loss {potential_loss:.4f} exceeds limit {max_trade_loss:.4f}",
                limit_type="per_trade_loss",
                current_value=potential_loss,
                limit_value=max_trade_loss,
            )

        return True

    def get_position_multiplier(self) -> float:
        """
        Get current position size multiplier based on drawdown state.

        Returns:
            Multiplier between 0 and 1.
        """
        if self._is_stopped or self._is_paused:
            return 0.0
        return self._position_multiplier

    def get_current_drawdown(self) -> float:
        """Get current drawdown from peak as percentage."""
        if self.peak_balance <= 0:
            return 0.0
        return (self.peak_balance - self.current_balance) / self.peak_balance

    def get_daily_drawdown(self) -> float:
        """Get today's drawdown as percentage of initial balance."""
        today = self._get_today()
        if today.pnl >= 0:
            return 0.0
        return abs(today.pnl) / self.initial_balance

    def get_stats(self) -> dict:
        """Get drawdown statistics."""
        today = self._get_today()

        return {
            "initial_balance": self.initial_balance,
            "current_balance": self.current_balance,
            "peak_balance": self.peak_balance,
            "current_drawdown": self.get_current_drawdown(),
            "current_drawdown_pct": f"{self.get_current_drawdown():.2%}",
            "daily_pnl": today.pnl,
            "daily_trades": today.trades,
            "daily_wins": today.wins,
            "daily_drawdown": self.get_daily_drawdown(),
            "position_multiplier": self._position_multiplier,
            "is_paused": self._is_paused,
            "is_stopped": self._is_stopped,
            "pause_until": self._pause_until,
        }

    def reset(self, new_balance: Optional[float] = None) -> None:
        """
        Reset drawdown controller.

        Args:
            new_balance: New starting balance (uses current if None).
        """
        if new_balance:
            self.initial_balance = new_balance
            self.current_balance = new_balance
            self.peak_balance = new_balance
        else:
            self.peak_balance = self.current_balance

        self._is_paused = False
        self._is_stopped = False
        self._pause_until = None
        self._position_multiplier = 1.0

        logger.info(f"DrawdownController reset: balance={self.current_balance}")

    def force_resume(self) -> None:
        """Force resume trading (manual override)."""
        self._is_paused = False
        self._is_stopped = False
        self._pause_until = None
        self._position_multiplier = 1.0
        logger.warning("Trading force resumed - manual override")
