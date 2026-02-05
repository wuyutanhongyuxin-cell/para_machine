"""
Helper functions for Paradex Trader.
"""

import hashlib
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from typing import Any, Optional, Union


def generate_trade_id(prefix: str = "T") -> str:
    """
    Generate a unique trade ID.

    Args:
        prefix: ID prefix.

    Returns:
        Unique trade ID (e.g., T-1234567890-abc123).
    """
    timestamp = int(time.time() * 1000)
    unique = uuid.uuid4().hex[:8]
    return f"{prefix}-{timestamp}-{unique}"


def generate_order_id(prefix: str = "O") -> str:
    """
    Generate a unique order ID.

    Args:
        prefix: ID prefix.

    Returns:
        Unique order ID.
    """
    return generate_trade_id(prefix)


def format_price(price: float, decimals: int = 2) -> str:
    """
    Format price for display.

    Args:
        price: Price value.
        decimals: Number of decimal places.

    Returns:
        Formatted price string.
    """
    return f"{price:,.{decimals}f}"


def format_size(size: float, decimals: int = 6) -> str:
    """
    Format position size for API.

    Args:
        size: Size value.
        decimals: Number of decimal places.

    Returns:
        Formatted size string.
    """
    return f"{size:.{decimals}f}"


def safe_float(value: Any, default: float = 0.0) -> float:
    """
    Safely convert value to float.

    Args:
        value: Value to convert.
        default: Default value if conversion fails.

    Returns:
        Float value.
    """
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    """
    Safely convert value to int.

    Args:
        value: Value to convert.
        default: Default value if conversion fails.

    Returns:
        Int value.
    """
    if value is None:
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def round_to_tick(price: float, tick_size: float, direction: str = "nearest") -> float:
    """
    Round price to nearest tick size.

    Args:
        price: Price to round.
        tick_size: Tick size (e.g., 0.01).
        direction: "nearest", "down", or "up".

    Returns:
        Rounded price.
    """
    decimal_price = Decimal(str(price))
    decimal_tick = Decimal(str(tick_size))

    if direction == "down":
        return float((decimal_price / decimal_tick).quantize(Decimal("1"), rounding=ROUND_DOWN) * decimal_tick)
    elif direction == "up":
        return float((decimal_price / decimal_tick).quantize(Decimal("1"), rounding=ROUND_UP) * decimal_tick)
    else:
        return float(round(decimal_price / decimal_tick) * decimal_tick)


def round_size(size: float, increment: float) -> float:
    """
    Round size to valid increment.

    Args:
        size: Size to round.
        increment: Size increment.

    Returns:
        Rounded size.
    """
    return round_to_tick(size, increment, direction="down")


def calculate_pnl(
    entry_price: float,
    exit_price: float,
    size: float,
    direction: str,
    fees: float = 0.0,
) -> tuple[float, float]:
    """
    Calculate PnL for a trade.

    Args:
        entry_price: Entry price.
        exit_price: Exit price.
        size: Position size.
        direction: "LONG" or "SHORT".
        fees: Total fees paid.

    Returns:
        Tuple of (pnl_amount, pnl_percentage).
    """
    if direction.upper() == "LONG":
        pnl = (exit_price - entry_price) * size
    else:
        pnl = (entry_price - exit_price) * size

    pnl -= fees

    pnl_pct = pnl / (entry_price * size) if entry_price * size > 0 else 0

    return pnl, pnl_pct


def calculate_position_value(price: float, size: float, leverage: int = 1) -> float:
    """
    Calculate position value.

    Args:
        price: Current price.
        size: Position size.
        leverage: Leverage used.

    Returns:
        Position value in quote currency.
    """
    return price * size / leverage


def timestamp_to_datetime(ts: Union[int, float]) -> datetime:
    """
    Convert Unix timestamp to datetime.

    Args:
        ts: Unix timestamp (seconds or milliseconds).

    Returns:
        Datetime object in UTC.
    """
    # Handle milliseconds
    if ts > 1e12:
        ts = ts / 1000

    return datetime.fromtimestamp(ts, tz=timezone.utc)


def datetime_to_timestamp(dt: datetime) -> float:
    """
    Convert datetime to Unix timestamp.

    Args:
        dt: Datetime object.

    Returns:
        Unix timestamp in seconds.
    """
    return dt.timestamp()


def now_timestamp() -> float:
    """
    Get current Unix timestamp.

    Returns:
        Current timestamp in seconds.
    """
    return time.time()


def now_utc() -> datetime:
    """
    Get current UTC datetime.

    Returns:
        Current datetime in UTC.
    """
    return datetime.now(timezone.utc)


def hash_dict(d: dict) -> str:
    """
    Create a hash of a dictionary.

    Args:
        d: Dictionary to hash.

    Returns:
        SHA256 hash string.
    """
    import json
    content = json.dumps(d, sort_keys=True, default=str)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def clamp(value: float, min_val: float, max_val: float) -> float:
    """
    Clamp value to range.

    Args:
        value: Value to clamp.
        min_val: Minimum value.
        max_val: Maximum value.

    Returns:
        Clamped value.
    """
    return max(min_val, min(max_val, value))


def exponential_backoff(
    attempt: int,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    factor: float = 2.0,
) -> float:
    """
    Calculate exponential backoff delay.

    Args:
        attempt: Current attempt number (0-indexed).
        base_delay: Base delay in seconds.
        max_delay: Maximum delay in seconds.
        factor: Exponential factor.

    Returns:
        Delay in seconds.
    """
    delay = base_delay * (factor ** attempt)
    return min(delay, max_delay)


def parse_market_symbol(market: str) -> tuple[str, str, str]:
    """
    Parse market symbol.

    Args:
        market: Market symbol (e.g., "BTC-USD-PERP").

    Returns:
        Tuple of (base, quote, type).
    """
    parts = market.split("-")
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    elif len(parts) == 2:
        return parts[0], parts[1], "SPOT"
    else:
        raise ValueError(f"Invalid market symbol: {market}")


def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format.

    Args:
        seconds: Duration in seconds.

    Returns:
        Formatted string (e.g., "1h 23m 45s").
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


def calculate_sharpe_ratio(
    returns: list[float],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 365,
) -> Optional[float]:
    """
    Calculate Sharpe ratio.

    Args:
        returns: List of returns.
        risk_free_rate: Annual risk-free rate.
        periods_per_year: Number of periods per year.

    Returns:
        Sharpe ratio or None if not enough data.
    """
    if len(returns) < 2:
        return None

    import numpy as np

    returns_arr = np.array(returns)
    excess_returns = returns_arr - (risk_free_rate / periods_per_year)

    mean_return = np.mean(excess_returns)
    std_return = np.std(excess_returns, ddof=1)

    if std_return == 0:
        return None

    return (mean_return / std_return) * np.sqrt(periods_per_year)


def calculate_max_drawdown(equity_curve: list[float]) -> tuple[float, int, int]:
    """
    Calculate maximum drawdown.

    Args:
        equity_curve: List of equity values.

    Returns:
        Tuple of (max_drawdown_pct, peak_index, trough_index).
    """
    if len(equity_curve) < 2:
        return 0.0, 0, 0

    import numpy as np

    equity = np.array(equity_curve)
    running_max = np.maximum.accumulate(equity)
    drawdowns = (equity - running_max) / running_max

    max_dd_idx = np.argmin(drawdowns)
    max_dd = drawdowns[max_dd_idx]

    # Find peak (most recent high before trough)
    peak_idx = np.argmax(equity[:max_dd_idx + 1])

    return abs(float(max_dd)), int(peak_idx), int(max_dd_idx)
