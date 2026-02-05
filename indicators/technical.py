"""
Technical Indicators for Paradex Trader.

Provides stateless calculation functions for common technical indicators.
All functions are designed to be memory-efficient and work with numpy arrays.
"""

import numpy as np
from typing import List, Optional, Tuple, Union

ArrayLike = Union[List[float], np.ndarray]


def ensure_array(data: ArrayLike) -> np.ndarray:
    """Convert input to numpy array."""
    if isinstance(data, np.ndarray):
        return data
    return np.array(data, dtype=np.float64)


def sma(data: ArrayLike, period: int) -> np.ndarray:
    """
    Simple Moving Average.

    Args:
        data: Price data.
        period: SMA period.

    Returns:
        SMA values (NaN for insufficient data).
    """
    arr = ensure_array(data)
    if len(arr) < period:
        return np.full(len(arr), np.nan)

    result = np.full(len(arr), np.nan)
    for i in range(period - 1, len(arr)):
        result[i] = np.mean(arr[i - period + 1:i + 1])
    return result


def ema(data: ArrayLike, period: int) -> np.ndarray:
    """
    Exponential Moving Average.

    Args:
        data: Price data.
        period: EMA period.

    Returns:
        EMA values.
    """
    arr = ensure_array(data)
    if len(arr) < period:
        return np.full(len(arr), np.nan)

    alpha = 2 / (period + 1)
    result = np.full(len(arr), np.nan)

    # Initialize with SMA
    result[period - 1] = np.mean(arr[:period])

    # Calculate EMA
    for i in range(period, len(arr)):
        result[i] = alpha * arr[i] + (1 - alpha) * result[i - 1]

    return result


def rsi(data: ArrayLike, period: int = 14) -> np.ndarray:
    """
    Relative Strength Index.

    Args:
        data: Price data (typically close prices).
        period: RSI period.

    Returns:
        RSI values (0-100).
    """
    arr = ensure_array(data)
    if len(arr) < period + 1:
        return np.full(len(arr), np.nan)

    # Calculate price changes
    changes = np.diff(arr)

    # Separate gains and losses
    gains = np.where(changes > 0, changes, 0)
    losses = np.where(changes < 0, -changes, 0)

    result = np.full(len(arr), np.nan)

    # Initial averages
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    if avg_loss == 0:
        result[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        result[period] = 100 - (100 / (1 + rs))

    # Wilder's smoothing
    for i in range(period + 1, len(arr)):
        avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period

        if avg_loss == 0:
            result[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[i] = 100 - (100 / (1 + rs))

    return result


def atr(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    period: int = 14
) -> np.ndarray:
    """
    Average True Range.

    Args:
        high: High prices.
        low: Low prices.
        close: Close prices.
        period: ATR period.

    Returns:
        ATR values.
    """
    high_arr = ensure_array(high)
    low_arr = ensure_array(low)
    close_arr = ensure_array(close)

    n = len(high_arr)
    if n < 2:
        return np.full(n, np.nan)

    # True Range
    tr = np.zeros(n)
    tr[0] = high_arr[0] - low_arr[0]

    for i in range(1, n):
        tr[i] = max(
            high_arr[i] - low_arr[i],
            abs(high_arr[i] - close_arr[i - 1]),
            abs(low_arr[i] - close_arr[i - 1])
        )

    # ATR using Wilder's smoothing
    result = np.full(n, np.nan)
    if n >= period:
        result[period - 1] = np.mean(tr[:period])

        for i in range(period, n):
            result[i] = (result[i - 1] * (period - 1) + tr[i]) / period

    return result


def bollinger_bands(
    data: ArrayLike,
    period: int = 20,
    num_std: float = 2.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bollinger Bands.

    Args:
        data: Price data.
        period: Moving average period.
        num_std: Number of standard deviations.

    Returns:
        Tuple of (upper_band, middle_band, lower_band).
    """
    arr = ensure_array(data)
    n = len(arr)

    middle = sma(arr, period)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)

    for i in range(period - 1, n):
        std = np.std(arr[i - period + 1:i + 1])
        upper[i] = middle[i] + num_std * std
        lower[i] = middle[i] - num_std * std

    return upper, middle, lower


def bollinger_position(
    price: float,
    upper: float,
    lower: float,
    middle: float
) -> float:
    """
    Calculate position within Bollinger Bands.

    Args:
        price: Current price.
        upper: Upper band.
        lower: Lower band.
        middle: Middle band.

    Returns:
        Position from -1 (lower band) to 1 (upper band).
    """
    if upper == lower:
        return 0.0

    band_width = (upper - lower) / 2
    if band_width == 0:
        return 0.0

    return (price - middle) / band_width


def donchian_channel(
    high: ArrayLike,
    low: ArrayLike,
    period: int = 20
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Donchian Channel.

    Args:
        high: High prices.
        low: Low prices.
        period: Lookback period.

    Returns:
        Tuple of (upper_channel, middle_channel, lower_channel).
    """
    high_arr = ensure_array(high)
    low_arr = ensure_array(low)
    n = len(high_arr)

    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    middle = np.full(n, np.nan)

    for i in range(period - 1, n):
        upper[i] = np.max(high_arr[i - period + 1:i + 1])
        lower[i] = np.min(low_arr[i - period + 1:i + 1])
        middle[i] = (upper[i] + lower[i]) / 2

    return upper, middle, lower


def macd(
    data: ArrayLike,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    MACD (Moving Average Convergence Divergence).

    Args:
        data: Price data.
        fast_period: Fast EMA period.
        slow_period: Slow EMA period.
        signal_period: Signal line period.

    Returns:
        Tuple of (macd_line, signal_line, histogram).
    """
    arr = ensure_array(data)

    fast_ema = ema(arr, fast_period)
    slow_ema = ema(arr, slow_period)

    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line[~np.isnan(macd_line)], signal_period)

    # Pad signal line
    pad_length = len(arr) - len(signal_line)
    signal_line = np.concatenate([np.full(pad_length, np.nan), signal_line])

    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def stochastic(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    k_period: int = 14,
    d_period: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stochastic Oscillator.

    Args:
        high: High prices.
        low: Low prices.
        close: Close prices.
        k_period: %K period.
        d_period: %D smoothing period.

    Returns:
        Tuple of (%K, %D).
    """
    high_arr = ensure_array(high)
    low_arr = ensure_array(low)
    close_arr = ensure_array(close)
    n = len(high_arr)

    k = np.full(n, np.nan)

    for i in range(k_period - 1, n):
        highest = np.max(high_arr[i - k_period + 1:i + 1])
        lowest = np.min(low_arr[i - k_period + 1:i + 1])

        if highest == lowest:
            k[i] = 50.0
        else:
            k[i] = 100 * (close_arr[i] - lowest) / (highest - lowest)

    d = sma(k[~np.isnan(k)], d_period)
    pad_length = n - len(d)
    d = np.concatenate([np.full(pad_length, np.nan), d])

    return k, d


def adx(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    period: int = 14
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Average Directional Index.

    Args:
        high: High prices.
        low: Low prices.
        close: Close prices.
        period: ADX period.

    Returns:
        Tuple of (ADX, +DI, -DI).
    """
    high_arr = ensure_array(high)
    low_arr = ensure_array(low)
    close_arr = ensure_array(close)
    n = len(high_arr)

    if n < period + 1:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)

    # Calculate +DM and -DM
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)

    for i in range(1, n):
        up_move = high_arr[i] - high_arr[i - 1]
        down_move = low_arr[i - 1] - low_arr[i]

        plus_dm[i] = up_move if up_move > down_move and up_move > 0 else 0
        minus_dm[i] = down_move if down_move > up_move and down_move > 0 else 0

        tr[i] = max(
            high_arr[i] - low_arr[i],
            abs(high_arr[i] - close_arr[i - 1]),
            abs(low_arr[i] - close_arr[i - 1])
        )

    # Smooth with Wilder's method
    atr_smooth = np.full(n, np.nan)
    plus_di = np.full(n, np.nan)
    minus_di = np.full(n, np.nan)
    adx_arr = np.full(n, np.nan)

    atr_smooth[period] = np.sum(tr[1:period + 1])
    plus_dm_smooth = np.sum(plus_dm[1:period + 1])
    minus_dm_smooth = np.sum(minus_dm[1:period + 1])

    for i in range(period, n):
        if i > period:
            atr_smooth[i] = atr_smooth[i - 1] - atr_smooth[i - 1] / period + tr[i]
            plus_dm_smooth = plus_dm_smooth - plus_dm_smooth / period + plus_dm[i]
            minus_dm_smooth = minus_dm_smooth - minus_dm_smooth / period + minus_dm[i]
        else:
            atr_smooth[i] = np.sum(tr[1:period + 1])

        if atr_smooth[i] > 0:
            plus_di[i] = 100 * plus_dm_smooth / atr_smooth[i]
            minus_di[i] = 100 * minus_dm_smooth / atr_smooth[i]

    # Calculate DX and ADX
    dx = np.full(n, np.nan)
    for i in range(period, n):
        if plus_di[i] + minus_di[i] > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])

    # ADX is smoothed DX
    adx_arr[2 * period - 1] = np.nanmean(dx[period:2 * period])
    for i in range(2 * period, n):
        if not np.isnan(adx_arr[i - 1]) and not np.isnan(dx[i]):
            adx_arr[i] = (adx_arr[i - 1] * (period - 1) + dx[i]) / period

    return adx_arr, plus_di, minus_di


def vwap(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    volume: ArrayLike
) -> np.ndarray:
    """
    Volume Weighted Average Price.

    Args:
        high: High prices.
        low: Low prices.
        close: Close prices.
        volume: Volume data.

    Returns:
        VWAP values.
    """
    high_arr = ensure_array(high)
    low_arr = ensure_array(low)
    close_arr = ensure_array(close)
    volume_arr = ensure_array(volume)

    typical_price = (high_arr + low_arr + close_arr) / 3
    cum_vol = np.cumsum(volume_arr)
    cum_tp_vol = np.cumsum(typical_price * volume_arr)

    return np.where(cum_vol > 0, cum_tp_vol / cum_vol, np.nan)


def momentum(data: ArrayLike, period: int = 10) -> np.ndarray:
    """
    Momentum indicator.

    Args:
        data: Price data.
        period: Lookback period.

    Returns:
        Momentum values.
    """
    arr = ensure_array(data)
    n = len(arr)

    result = np.full(n, np.nan)
    for i in range(period, n):
        result[i] = arr[i] - arr[i - period]

    return result


def roc(data: ArrayLike, period: int = 10) -> np.ndarray:
    """
    Rate of Change.

    Args:
        data: Price data.
        period: Lookback period.

    Returns:
        ROC values (percentage).
    """
    arr = ensure_array(data)
    n = len(arr)

    result = np.full(n, np.nan)
    for i in range(period, n):
        if arr[i - period] != 0:
            result[i] = ((arr[i] - arr[i - period]) / arr[i - period]) * 100

    return result


def williams_r(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    period: int = 14
) -> np.ndarray:
    """
    Williams %R.

    Args:
        high: High prices.
        low: Low prices.
        close: Close prices.
        period: Lookback period.

    Returns:
        Williams %R values (-100 to 0).
    """
    high_arr = ensure_array(high)
    low_arr = ensure_array(low)
    close_arr = ensure_array(close)
    n = len(high_arr)

    result = np.full(n, np.nan)

    for i in range(period - 1, n):
        highest = np.max(high_arr[i - period + 1:i + 1])
        lowest = np.min(low_arr[i - period + 1:i + 1])

        if highest == lowest:
            result[i] = -50.0
        else:
            result[i] = -100 * (highest - close_arr[i]) / (highest - lowest)

    return result


class TechnicalIndicators:
    """
    Technical indicators calculator with caching.

    Provides a convenient interface for calculating multiple indicators
    on the same price data.
    """

    def __init__(self):
        """Initialize calculator."""
        self._cache = {}

    def clear_cache(self) -> None:
        """Clear indicator cache."""
        self._cache.clear()

    def calculate_all(
        self,
        high: ArrayLike,
        low: ArrayLike,
        close: ArrayLike,
        volume: Optional[ArrayLike] = None
    ) -> dict:
        """
        Calculate all indicators.

        Args:
            high: High prices.
            low: Low prices.
            close: Close prices.
            volume: Volume data (optional).

        Returns:
            Dictionary of indicator values.
        """
        results = {}

        # Trend indicators
        results["sma_20"] = sma(close, 20)
        results["ema_12"] = ema(close, 12)
        results["ema_26"] = ema(close, 26)

        # Momentum indicators
        results["rsi_14"] = rsi(close, 14)
        results["momentum_10"] = momentum(close, 10)
        results["roc_10"] = roc(close, 10)

        # Volatility indicators
        results["atr_14"] = atr(high, low, close, 14)
        bb_upper, bb_middle, bb_lower = bollinger_bands(close, 20, 2.0)
        results["bb_upper"] = bb_upper
        results["bb_middle"] = bb_middle
        results["bb_lower"] = bb_lower

        # Channel indicators
        dc_upper, dc_middle, dc_lower = donchian_channel(high, low, 20)
        results["dc_upper"] = dc_upper
        results["dc_middle"] = dc_middle
        results["dc_lower"] = dc_lower

        # MACD
        macd_line, signal_line, histogram = macd(close)
        results["macd_line"] = macd_line
        results["macd_signal"] = signal_line
        results["macd_histogram"] = histogram

        # Stochastic
        k, d = stochastic(high, low, close)
        results["stoch_k"] = k
        results["stoch_d"] = d

        # ADX
        adx_val, plus_di, minus_di = adx(high, low, close)
        results["adx"] = adx_val
        results["plus_di"] = plus_di
        results["minus_di"] = minus_di

        # Williams %R
        results["williams_r"] = williams_r(high, low, close)

        # VWAP (if volume available)
        if volume is not None:
            results["vwap"] = vwap(high, low, close, volume)

        return results
