"""
Performance metrics calculation for Paradex Trader.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np


@dataclass
class TradeResult:
    """Single trade result for metrics calculation."""

    pnl: float
    pnl_pct: float
    is_win: bool
    hold_time: float
    timestamp: datetime
    strategy: str


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""

    # Basic stats
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0

    # PnL metrics
    total_pnl: float = 0.0
    total_fees: float = 0.0
    net_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0

    # Ratios
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_win_loss_ratio: float = 0.0
    expectancy: float = 0.0

    # Risk metrics
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    calmar_ratio: Optional[float] = None

    # Time metrics
    avg_hold_time: float = 0.0
    max_hold_time: float = 0.0
    min_hold_time: float = 0.0

    # Streak metrics
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    current_streak: int = 0

    # Period
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None


class MetricsCalculator:
    """Calculate trading performance metrics."""

    def __init__(self, initial_capital: float = 100.0):
        """
        Initialize metrics calculator.

        Args:
            initial_capital: Initial capital for calculations.
        """
        self.initial_capital = initial_capital
        self.trades: List[TradeResult] = []
        self.equity_curve: List[float] = [initial_capital]

    def add_trade(self, trade: TradeResult) -> None:
        """
        Add a trade result.

        Args:
            trade: Trade result to add.
        """
        self.trades.append(trade)
        self.equity_curve.append(self.equity_curve[-1] + trade.pnl)

    def calculate(self) -> PerformanceMetrics:
        """
        Calculate all performance metrics.

        Returns:
            PerformanceMetrics object with all calculated metrics.
        """
        metrics = PerformanceMetrics()

        if not self.trades:
            return metrics

        # Basic counts
        metrics.total_trades = len(self.trades)
        metrics.winning_trades = sum(1 for t in self.trades if t.is_win)
        metrics.losing_trades = metrics.total_trades - metrics.winning_trades

        # PnL calculations
        pnls = [t.pnl for t in self.trades]
        winning_pnls = [t.pnl for t in self.trades if t.is_win]
        losing_pnls = [t.pnl for t in self.trades if not t.is_win]

        metrics.total_pnl = sum(pnls)
        metrics.gross_profit = sum(winning_pnls) if winning_pnls else 0.0
        metrics.gross_loss = abs(sum(losing_pnls)) if losing_pnls else 0.0
        metrics.net_pnl = metrics.total_pnl - metrics.total_fees

        # Win rate
        metrics.win_rate = metrics.winning_trades / metrics.total_trades

        # Average win/loss
        metrics.avg_win = np.mean(winning_pnls) if winning_pnls else 0.0
        metrics.avg_loss = abs(np.mean(losing_pnls)) if losing_pnls else 0.0

        # Win/loss ratio
        if metrics.avg_loss > 0:
            metrics.avg_win_loss_ratio = metrics.avg_win / metrics.avg_loss
        else:
            metrics.avg_win_loss_ratio = float("inf") if metrics.avg_win > 0 else 0.0

        # Profit factor
        if metrics.gross_loss > 0:
            metrics.profit_factor = metrics.gross_profit / metrics.gross_loss
        else:
            metrics.profit_factor = float("inf") if metrics.gross_profit > 0 else 0.0

        # Expectancy
        metrics.expectancy = (
            metrics.win_rate * metrics.avg_win -
            (1 - metrics.win_rate) * metrics.avg_loss
        )

        # Time metrics
        hold_times = [t.hold_time for t in self.trades]
        metrics.avg_hold_time = np.mean(hold_times)
        metrics.max_hold_time = max(hold_times)
        metrics.min_hold_time = min(hold_times)

        # Streak metrics
        self._calculate_streaks(metrics)

        # Risk metrics
        self._calculate_risk_metrics(metrics)

        # Period
        timestamps = [t.timestamp for t in self.trades]
        metrics.period_start = min(timestamps)
        metrics.period_end = max(timestamps)

        return metrics

    def _calculate_streaks(self, metrics: PerformanceMetrics) -> None:
        """Calculate win/loss streaks."""
        if not self.trades:
            return

        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0

        for trade in self.trades:
            if trade.is_win:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)

        metrics.max_consecutive_wins = max_wins
        metrics.max_consecutive_losses = max_losses

        # Current streak
        if self.trades[-1].is_win:
            metrics.current_streak = current_wins
        else:
            metrics.current_streak = -current_losses

    def _calculate_risk_metrics(self, metrics: PerformanceMetrics) -> None:
        """Calculate risk-adjusted metrics."""
        if len(self.equity_curve) < 2:
            return

        equity = np.array(self.equity_curve)

        # Max drawdown
        running_max = np.maximum.accumulate(equity)
        drawdowns = (equity - running_max) / running_max
        metrics.max_drawdown_pct = abs(float(np.min(drawdowns)))
        metrics.max_drawdown = metrics.max_drawdown_pct * self.initial_capital

        # Calculate returns
        returns = np.diff(equity) / equity[:-1]

        if len(returns) < 2:
            return

        # Sharpe ratio (assuming 0 risk-free rate)
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)

        if std_return > 0:
            # Annualize assuming daily trades
            metrics.sharpe_ratio = float(mean_return / std_return * np.sqrt(365))

        # Sortino ratio (downside deviation)
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0:
            downside_std = np.std(negative_returns, ddof=1)
            if downside_std > 0:
                metrics.sortino_ratio = float(mean_return / downside_std * np.sqrt(365))

        # Calmar ratio
        if metrics.max_drawdown_pct > 0:
            # Annualized return / max drawdown
            total_return = (equity[-1] - equity[0]) / equity[0]
            days = (self.trades[-1].timestamp - self.trades[0].timestamp).days or 1
            annualized_return = total_return * (365 / days)
            metrics.calmar_ratio = float(annualized_return / metrics.max_drawdown_pct)

    def get_equity_curve(self) -> List[float]:
        """Get equity curve."""
        return self.equity_curve.copy()

    def get_drawdown_curve(self) -> List[float]:
        """Get drawdown curve."""
        equity = np.array(self.equity_curve)
        running_max = np.maximum.accumulate(equity)
        drawdowns = (equity - running_max) / running_max
        return drawdowns.tolist()

    def get_strategy_breakdown(self) -> dict[str, PerformanceMetrics]:
        """
        Get metrics breakdown by strategy.

        Returns:
            Dictionary mapping strategy names to their metrics.
        """
        strategies = set(t.strategy for t in self.trades)
        breakdown = {}

        for strategy in strategies:
            strategy_trades = [t for t in self.trades if t.strategy == strategy]

            calc = MetricsCalculator(self.initial_capital)
            for trade in strategy_trades:
                calc.add_trade(trade)

            breakdown[strategy] = calc.calculate()

        return breakdown

    def get_daily_pnl(self) -> dict[str, float]:
        """
        Get PnL grouped by date.

        Returns:
            Dictionary mapping date strings to daily PnL.
        """
        daily = {}

        for trade in self.trades:
            date_key = trade.timestamp.strftime("%Y-%m-%d")
            daily[date_key] = daily.get(date_key, 0.0) + trade.pnl

        return daily

    def reset(self) -> None:
        """Reset all data."""
        self.trades.clear()
        self.equity_curve = [self.initial_capital]


class RollingMetrics:
    """Calculate rolling performance metrics over a window."""

    def __init__(self, window_size: int = 50):
        """
        Initialize rolling metrics calculator.

        Args:
            window_size: Number of trades in rolling window.
        """
        self.window_size = window_size
        self.trades: List[TradeResult] = []

    def add_trade(self, trade: TradeResult) -> None:
        """Add a trade and maintain window size."""
        self.trades.append(trade)
        if len(self.trades) > self.window_size:
            self.trades.pop(0)

    def get_win_rate(self) -> float:
        """Get rolling win rate."""
        if not self.trades:
            return 0.0
        wins = sum(1 for t in self.trades if t.is_win)
        return wins / len(self.trades)

    def get_avg_pnl(self) -> float:
        """Get rolling average PnL."""
        if not self.trades:
            return 0.0
        return np.mean([t.pnl for t in self.trades])

    def get_volatility(self) -> float:
        """Get rolling PnL volatility."""
        if len(self.trades) < 2:
            return 0.0
        return float(np.std([t.pnl for t in self.trades]))

    def get_sharpe(self) -> Optional[float]:
        """Get rolling Sharpe ratio."""
        if len(self.trades) < 10:
            return None

        pnls = [t.pnl for t in self.trades]
        mean_pnl = np.mean(pnls)
        std_pnl = np.std(pnls, ddof=1)

        if std_pnl == 0:
            return None

        return float(mean_pnl / std_pnl)

    def is_performing_well(self, min_win_rate: float = 0.4, min_expectancy: float = 0.0) -> bool:
        """
        Check if recent performance meets thresholds.

        Args:
            min_win_rate: Minimum acceptable win rate.
            min_expectancy: Minimum acceptable expectancy.

        Returns:
            True if performance is acceptable.
        """
        if len(self.trades) < 10:
            return True  # Not enough data

        win_rate = self.get_win_rate()
        avg_pnl = self.get_avg_pnl()

        return win_rate >= min_win_rate and avg_pnl >= min_expectancy
