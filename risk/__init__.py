"""Risk management module."""

from .position_sizer import PositionSizer, PositionSizeResult
from .drawdown_control import DrawdownController, DrawdownLevel, DailyPnL
from .cooldown import CooldownManager, CooldownEvent

__all__ = [
    # Position sizing
    "PositionSizer",
    "PositionSizeResult",
    # Drawdown control
    "DrawdownController",
    "DrawdownLevel",
    "DailyPnL",
    # Cooldown
    "CooldownManager",
    "CooldownEvent",
]


class RiskManager:
    """
    Unified risk management interface.

    Combines position sizing, drawdown control, and cooldown management
    into a single coordinator.
    """

    def __init__(
        self,
        initial_balance: float,
        base_trade_size: float = 0.001,
        max_trade_size: float = 0.005,
        max_daily_loss_pct: float = 0.05,
        max_total_drawdown_pct: float = 0.15,
        max_consecutive_losses: int = 5,
        cooldown_seconds: int = 300,
        use_kelly: bool = True,
    ):
        """
        Initialize risk manager.

        Args:
            initial_balance: Starting account balance.
            base_trade_size: Base position size.
            max_trade_size: Maximum position size.
            max_daily_loss_pct: Maximum daily loss percentage.
            max_total_drawdown_pct: Maximum total drawdown.
            max_consecutive_losses: Consecutive losses before cooldown.
            cooldown_seconds: Default cooldown duration.
            use_kelly: Whether to use Kelly criterion.
        """
        self.position_sizer = PositionSizer(
            base_size=base_trade_size,
            max_size=max_trade_size,
            use_kelly=use_kelly,
        )

        self.drawdown_controller = DrawdownController(
            initial_balance=initial_balance,
            max_daily_loss_pct=max_daily_loss_pct,
            max_total_drawdown_pct=max_total_drawdown_pct,
        )

        self.cooldown_manager = CooldownManager(
            max_consecutive_losses=max_consecutive_losses,
            cooldown_seconds=cooldown_seconds,
        )

    def can_trade(self, strategy: str = None) -> tuple[bool, str]:
        """
        Check if trading is allowed.

        Args:
            strategy: Strategy to check.

        Returns:
            Tuple of (can_trade, reason).
        """
        # Check drawdown
        can_trade, reason = self.drawdown_controller.can_trade()
        if not can_trade:
            return False, reason

        # Check cooldown
        in_cooldown, reason, _ = self.cooldown_manager.is_in_cooldown(strategy)
        if in_cooldown:
            return False, reason

        return True, "ok"

    def calculate_position_size(
        self,
        balance: float,
        price: float,
        stop_loss_pct: float,
        signal_strength: float = 1.0,
        volatility: float = 0.02,
        strategy_multiplier: float = 1.0,
    ) -> PositionSizeResult:
        """Calculate position size with risk adjustments."""
        # Get drawdown multiplier
        dd_mult = self.drawdown_controller.get_position_multiplier()

        # Check if should reduce risk
        should_reduce, cooldown_mult = self.cooldown_manager.should_reduce_risk()

        # Combine multipliers
        total_mult = strategy_multiplier * dd_mult * cooldown_mult

        return self.position_sizer.calculate_size(
            balance=balance,
            price=price,
            stop_loss_pct=stop_loss_pct,
            signal_strength=signal_strength,
            volatility=volatility,
            strategy_multiplier=total_mult,
        )

    def record_trade_result(
        self,
        pnl: float,
        pnl_pct: float,
        is_win: bool,
        strategy: str = None,
    ) -> None:
        """
        Record trade result across all risk components.

        Args:
            pnl: Profit/loss amount.
            pnl_pct: Profit/loss percentage.
            is_win: Whether trade was profitable.
            strategy: Strategy that made the trade.
        """
        # Update position sizer
        self.position_sizer.update_performance(pnl, is_win)

        # Update drawdown controller
        self.drawdown_controller.record_trade(pnl, is_win)

        # Update cooldown manager
        self.cooldown_manager.record_trade_result(is_win, pnl_pct, strategy)

    def update_balance(self, new_balance: float) -> None:
        """Update current balance."""
        self.drawdown_controller.update_balance(new_balance)

    def get_stats(self) -> dict:
        """Get combined risk statistics."""
        return {
            "position_sizer": self.position_sizer.get_stats(),
            "drawdown": self.drawdown_controller.get_stats(),
            "cooldown": self.cooldown_manager.get_stats(),
        }

    def reset(self, new_balance: float = None) -> None:
        """Reset all risk components."""
        self.position_sizer.reset_performance()
        self.drawdown_controller.reset(new_balance)
        self.cooldown_manager.clear_all_cooldowns()
        self.cooldown_manager.reset_consecutive_losses()
