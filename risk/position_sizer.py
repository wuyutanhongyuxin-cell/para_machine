"""
Position Sizing module for Paradex Trader.

Implements multiple position sizing methods:
- Fixed size
- Kelly Criterion
- ATR-based
- Volatility-adjusted

All methods respect maximum position limits and account risk constraints.
"""

import logging
import math
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("paradex_trader.risk.position_sizer")


@dataclass
class PositionSizeResult:
    """Result of position size calculation."""
    size: float
    method: str
    raw_size: float
    multiplier: float
    capped: bool
    reason: str

    def to_dict(self) -> dict:
        return {
            "size": self.size,
            "method": self.method,
            "raw_size": self.raw_size,
            "multiplier": self.multiplier,
            "capped": self.capped,
            "reason": self.reason,
        }


class PositionSizer:
    """
    Position sizing calculator.

    Determines optimal position size based on:
    - Account balance and risk tolerance
    - Signal strength
    - Market volatility
    - Kelly criterion (optional)
    """

    def __init__(
        self,
        base_size: float = 0.001,
        max_size: float = 0.005,
        min_size: float = 0.0001,
        use_kelly: bool = True,
        kelly_fraction: float = 0.25,
        min_multiplier: float = 0.5,
        max_multiplier: float = 2.0,
        max_risk_per_trade: float = 0.03,
    ):
        """
        Initialize position sizer.

        Args:
            base_size: Base position size (e.g., 0.001 BTC).
            max_size: Maximum position size.
            min_size: Minimum position size.
            use_kelly: Whether to use Kelly criterion.
            kelly_fraction: Fraction of Kelly to use (quarter-Kelly = 0.25).
            min_multiplier: Minimum size multiplier.
            max_multiplier: Maximum size multiplier.
            max_risk_per_trade: Maximum risk per trade as fraction of balance.
        """
        self.base_size = base_size
        self.max_size = max_size
        self.min_size = min_size
        self.use_kelly = use_kelly
        self.kelly_fraction = kelly_fraction
        self.min_multiplier = min_multiplier
        self.max_multiplier = max_multiplier
        self.max_risk_per_trade = max_risk_per_trade

        # Performance tracking for Kelly
        self._wins = 0
        self._losses = 0
        self._total_win_amount = 0.0
        self._total_loss_amount = 0.0

        logger.info(
            f"PositionSizer initialized: base={base_size}, max={max_size}, "
            f"kelly={use_kelly}, kelly_fraction={kelly_fraction}"
        )

    def calculate_size(
        self,
        balance: float,
        price: float,
        stop_loss_pct: float,
        signal_strength: float = 1.0,
        volatility: float = 0.02,
        strategy_multiplier: float = 1.0,
    ) -> PositionSizeResult:
        """
        Calculate optimal position size.

        Args:
            balance: Account balance in USDC.
            price: Current asset price.
            stop_loss_pct: Stop loss percentage (e.g., 0.02 for 2%).
            signal_strength: Signal strength from 0 to 1.
            volatility: Current volatility.
            strategy_multiplier: Strategy-specific multiplier.

        Returns:
            PositionSizeResult with calculated size.
        """
        # Start with base size
        raw_size = self.base_size
        method = "fixed"
        multiplier = 1.0

        # Calculate maximum size based on risk
        max_risk_amount = balance * self.max_risk_per_trade
        if stop_loss_pct > 0:
            risk_based_size = max_risk_amount / (price * stop_loss_pct)
        else:
            risk_based_size = self.max_size

        # Kelly criterion sizing
        if self.use_kelly and self._has_enough_data():
            kelly_size = self._calculate_kelly_size(balance, price)
            if kelly_size > 0:
                raw_size = kelly_size
                method = "kelly"

        # Apply signal strength multiplier
        signal_mult = self._signal_to_multiplier(signal_strength)
        multiplier *= signal_mult

        # Apply volatility adjustment
        vol_mult = self._volatility_multiplier(volatility)
        multiplier *= vol_mult

        # Apply strategy multiplier
        multiplier *= strategy_multiplier

        # Clamp multiplier
        multiplier = max(self.min_multiplier, min(self.max_multiplier, multiplier))

        # Calculate final size
        final_size = raw_size * multiplier

        # Apply limits
        capped = False
        cap_reason = ""

        # Risk-based cap
        if final_size > risk_based_size:
            final_size = risk_based_size
            capped = True
            cap_reason = "risk_limit"

        # Maximum size cap
        if final_size > self.max_size:
            final_size = self.max_size
            capped = True
            cap_reason = "max_size"

        # Minimum size
        if final_size < self.min_size:
            final_size = self.min_size
            capped = True
            cap_reason = "min_size"

        # Position value check
        position_value = final_size * price
        if position_value > balance * 0.5:  # Don't use more than 50% of balance
            final_size = (balance * 0.5) / price
            capped = True
            cap_reason = "balance_limit"

        result = PositionSizeResult(
            size=round(final_size, 6),
            method=method,
            raw_size=raw_size,
            multiplier=multiplier,
            capped=capped,
            reason=cap_reason if capped else "normal",
        )

        logger.debug(
            f"Position size calculated: {result.size} ({method}), "
            f"mult={multiplier:.2f}, capped={capped}"
        )

        return result

    def _calculate_kelly_size(self, balance: float, price: float) -> float:
        """
        Calculate Kelly criterion position size.

        Kelly formula: f* = (p * b - q) / b
        where:
            p = win probability
            b = win/loss ratio
            q = loss probability (1 - p)
        """
        if self._wins + self._losses < 10:
            return 0  # Not enough data

        win_rate = self._wins / (self._wins + self._losses)
        loss_rate = 1 - win_rate

        if self._losses == 0 or self._total_loss_amount == 0:
            return 0

        avg_win = self._total_win_amount / max(self._wins, 1)
        avg_loss = self._total_loss_amount / max(self._losses, 1)

        if avg_loss == 0:
            return 0

        win_loss_ratio = avg_win / avg_loss

        # Kelly formula
        kelly = (win_rate * win_loss_ratio - loss_rate) / win_loss_ratio

        # Apply fraction (quarter-Kelly by default)
        kelly *= self.kelly_fraction

        # Kelly can be negative (don't trade)
        if kelly <= 0:
            return 0

        # Convert to position size
        # Kelly gives fraction of bankroll to risk
        risk_amount = balance * kelly
        position_value = risk_amount / 0.02  # Assume 2% stop
        position_size = position_value / price

        return position_size

    def _signal_to_multiplier(self, signal_strength: float) -> float:
        """
        Convert signal strength to size multiplier.

        Mapping:
            0.4 -> 0.7
            0.6 -> 1.0
            0.8 -> 1.2
            1.0 -> 1.4
        """
        # Linear interpolation
        return 0.4 + signal_strength * 1.0

    def _volatility_multiplier(self, volatility: float) -> float:
        """
        Adjust size based on volatility.

        Lower volatility -> larger position (more predictable)
        Higher volatility -> smaller position (more risk)
        """
        # Target volatility around 2%
        target_vol = 0.02

        if volatility <= 0:
            return 1.0

        ratio = target_vol / volatility
        # Clamp to reasonable range
        return max(0.5, min(1.5, ratio))

    def _has_enough_data(self) -> bool:
        """Check if we have enough data for Kelly calculation."""
        return self._wins + self._losses >= 20

    def update_performance(self, pnl: float, is_win: bool) -> None:
        """
        Update performance tracking for Kelly calculation.

        Args:
            pnl: Profit/loss amount.
            is_win: Whether trade was profitable.
        """
        if is_win:
            self._wins += 1
            self._total_win_amount += abs(pnl)
        else:
            self._losses += 1
            self._total_loss_amount += abs(pnl)

        logger.debug(
            f"Performance updated: wins={self._wins}, losses={self._losses}, "
            f"total_win={self._total_win_amount:.4f}, total_loss={self._total_loss_amount:.4f}"
        )

    def get_kelly_fraction(self) -> Optional[float]:
        """
        Get current Kelly fraction recommendation.

        Returns:
            Kelly fraction or None if not enough data.
        """
        if not self._has_enough_data():
            return None

        win_rate = self._wins / (self._wins + self._losses)

        if self._losses == 0 or self._total_loss_amount == 0:
            return None

        avg_win = self._total_win_amount / self._wins
        avg_loss = self._total_loss_amount / self._losses
        win_loss_ratio = avg_win / avg_loss

        kelly = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio

        return kelly if kelly > 0 else 0

    def get_stats(self) -> dict:
        """Get position sizer statistics."""
        kelly = self.get_kelly_fraction()

        return {
            "base_size": self.base_size,
            "max_size": self.max_size,
            "use_kelly": self.use_kelly,
            "kelly_fraction_config": self.kelly_fraction,
            "kelly_fraction_actual": kelly,
            "wins": self._wins,
            "losses": self._losses,
            "total_trades": self._wins + self._losses,
            "win_rate": self._wins / (self._wins + self._losses) if self._wins + self._losses > 0 else 0,
        }

    def reset_performance(self) -> None:
        """Reset performance tracking."""
        self._wins = 0
        self._losses = 0
        self._total_win_amount = 0.0
        self._total_loss_amount = 0.0
        logger.info("Position sizer performance reset")
