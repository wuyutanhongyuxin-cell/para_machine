"""
Volatility Indicators for Paradex Trader.

Provides various volatility measurement methods:
- Historical volatility
- Parkinson volatility
- Garman-Klass volatility
- Yang-Zhang volatility
- Volatility percentile ranking
"""

import numpy as np
from collections import deque
from typing import List, Optional, Tuple, Deque, Union

ArrayLike = Union[List[float], np.ndarray]


def ensure_array(data: ArrayLike) -> np.ndarray:
    """Convert input to numpy array."""
    if isinstance(data, np.ndarray):
        return data
    return np.array(data, dtype=np.float64)


def historical_volatility(
    close: ArrayLike,
    period: int = 20,
    annualize: bool = False,
    trading_periods: int = 365,
) -> np.ndarray:
    """
    Calculate historical (realized) volatility.

    Standard deviation of log returns.

    Args:
        close: Close prices.
        period: Lookback period.
        annualize: Whether to annualize the volatility.
        trading_periods: Number of trading periods per year.

    Returns:
        Volatility values.
    """
    arr = ensure_array(close)
    n = len(arr)

    if n < 2:
        return np.full(n, np.nan)

    # Log returns
    log_returns = np.log(arr[1:] / arr[:-1])

    result = np.full(n, np.nan)

    for i in range(period, n):
        window = log_returns[i - period:i]
        vol = np.std(window, ddof=1)

        if annualize:
            vol *= np.sqrt(trading_periods)

        result[i] = vol

    return result


def parkinson_volatility(
    high: ArrayLike,
    low: ArrayLike,
    period: int = 20,
    annualize: bool = False,
    trading_periods: int = 365,
) -> np.ndarray:
    """
    Parkinson volatility estimator.

    Uses high-low range, more efficient than close-to-close.
    Assumes no drift and continuous trading.

    Args:
        high: High prices.
        low: Low prices.
        period: Lookback period.
        annualize: Whether to annualize.
        trading_periods: Trading periods per year.

    Returns:
        Volatility values.
    """
    high_arr = ensure_array(high)
    low_arr = ensure_array(low)
    n = len(high_arr)

    if n < 1:
        return np.full(n, np.nan)

    # Log high/low
    log_hl = np.log(high_arr / low_arr) ** 2

    result = np.full(n, np.nan)
    factor = 1 / (4 * np.log(2))

    for i in range(period - 1, n):
        window = log_hl[i - period + 1:i + 1]
        vol = np.sqrt(factor * np.mean(window))

        if annualize:
            vol *= np.sqrt(trading_periods)

        result[i] = vol

    return result


def garman_klass_volatility(
    open_: ArrayLike,
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    period: int = 20,
    annualize: bool = False,
    trading_periods: int = 365,
) -> np.ndarray:
    """
    Garman-Klass volatility estimator.

    Uses OHLC data, more efficient than Parkinson.

    Args:
        open_: Open prices.
        high: High prices.
        low: Low prices.
        close: Close prices.
        period: Lookback period.
        annualize: Whether to annualize.
        trading_periods: Trading periods per year.

    Returns:
        Volatility values.
    """
    open_arr = ensure_array(open_)
    high_arr = ensure_array(high)
    low_arr = ensure_array(low)
    close_arr = ensure_array(close)
    n = len(high_arr)

    if n < 1:
        return np.full(n, np.nan)

    log_hl = np.log(high_arr / low_arr) ** 2
    log_co = np.log(close_arr / open_arr) ** 2

    result = np.full(n, np.nan)

    for i in range(period - 1, n):
        hl_window = log_hl[i - period + 1:i + 1]
        co_window = log_co[i - period + 1:i + 1]

        vol = np.sqrt(0.5 * np.mean(hl_window) - (2 * np.log(2) - 1) * np.mean(co_window))

        if annualize:
            vol *= np.sqrt(trading_periods)

        result[i] = max(0, vol)  # Ensure non-negative

    return result


def yang_zhang_volatility(
    open_: ArrayLike,
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    period: int = 20,
    annualize: bool = False,
    trading_periods: int = 365,
) -> np.ndarray:
    """
    Yang-Zhang volatility estimator.

    Handles overnight jumps, most accurate for daily data.

    Args:
        open_: Open prices.
        high: High prices.
        low: Low prices.
        close: Close prices.
        period: Lookback period.
        annualize: Whether to annualize.
        trading_periods: Trading periods per year.

    Returns:
        Volatility values.
    """
    open_arr = ensure_array(open_)
    high_arr = ensure_array(high)
    low_arr = ensure_array(low)
    close_arr = ensure_array(close)
    n = len(high_arr)

    if n < 2:
        return np.full(n, np.nan)

    result = np.full(n, np.nan)

    for i in range(period, n):
        # Overnight volatility (close to open)
        log_oc = np.log(open_arr[i - period + 1:i + 1] / close_arr[i - period:i])
        overnight_var = np.var(log_oc, ddof=1)

        # Open to close volatility
        log_co = np.log(close_arr[i - period + 1:i + 1] / open_arr[i - period + 1:i + 1])
        open_close_var = np.var(log_co, ddof=1)

        # Rogers-Satchell volatility
        log_ho = np.log(high_arr[i - period + 1:i + 1] / open_arr[i - period + 1:i + 1])
        log_lo = np.log(low_arr[i - period + 1:i + 1] / open_arr[i - period + 1:i + 1])
        log_hc = np.log(high_arr[i - period + 1:i + 1] / close_arr[i - period + 1:i + 1])
        log_lc = np.log(low_arr[i - period + 1:i + 1] / close_arr[i - period + 1:i + 1])

        rs_var = np.mean(log_ho * log_hc + log_lo * log_lc)

        # Combine
        k = 0.34 / (1.34 + (period + 1) / (period - 1))
        vol = np.sqrt(overnight_var + k * open_close_var + (1 - k) * rs_var)

        if annualize:
            vol *= np.sqrt(trading_periods)

        result[i] = vol

    return result


class VolatilityIndicators:
    """
    Volatility indicators calculator with history tracking.

    Provides:
    - Multiple volatility estimators
    - Volatility percentile ranking
    - Volatility regime detection
    - Volatility forecasting
    """

    def __init__(self, history_size: int = 500):
        """
        Initialize volatility calculator.

        Args:
            history_size: Number of historical values to track.
        """
        self.history_size = history_size

        # Price history for calculations
        self.open_history: Deque[float] = deque(maxlen=history_size)
        self.high_history: Deque[float] = deque(maxlen=history_size)
        self.low_history: Deque[float] = deque(maxlen=history_size)
        self.close_history: Deque[float] = deque(maxlen=history_size)

        # Volatility history
        self.vol_history: Deque[float] = deque(maxlen=history_size)

    def update(
        self,
        open_: float,
        high: float,
        low: float,
        close: float,
    ) -> dict:
        """
        Update with new OHLC data.

        Args:
            open_: Open price.
            high: High price.
            low: Low price.
            close: Close price.

        Returns:
            Dictionary of volatility metrics.
        """
        self.open_history.append(open_)
        self.high_history.append(high)
        self.low_history.append(low)
        self.close_history.append(close)

        metrics = self.calculate_all()

        if "historical_20" in metrics and not np.isnan(metrics["historical_20"]):
            self.vol_history.append(metrics["historical_20"])

        return metrics

    def calculate_all(self, period: int = 20) -> dict:
        """
        Calculate all volatility metrics.

        Args:
            period: Calculation period.

        Returns:
            Dictionary of volatility values.
        """
        metrics = {}

        if len(self.close_history) < period:
            return metrics

        close = list(self.close_history)
        high = list(self.high_history)
        low = list(self.low_history)
        open_ = list(self.open_history)

        # Historical volatility
        hv = historical_volatility(close, period)
        if len(hv) > 0 and not np.isnan(hv[-1]):
            metrics["historical_20"] = hv[-1]

        # Parkinson
        pv = parkinson_volatility(high, low, period)
        if len(pv) > 0 and not np.isnan(pv[-1]):
            metrics["parkinson_20"] = pv[-1]

        # Garman-Klass
        gk = garman_klass_volatility(open_, high, low, close, period)
        if len(gk) > 0 and not np.isnan(gk[-1]):
            metrics["garman_klass_20"] = gk[-1]

        # Percentile
        metrics["vol_percentile"] = self.get_volatility_percentile()

        # Regime
        metrics["vol_regime"] = self.get_volatility_regime()

        # Trend
        metrics["vol_trend"] = self.get_volatility_trend()

        return metrics

    def get_current_volatility(self, method: str = "historical") -> float:
        """
        Get current volatility using specified method.

        Args:
            method: "historical", "parkinson", or "garman_klass".

        Returns:
            Current volatility value.
        """
        if len(self.close_history) < 20:
            return 0.0

        close = list(self.close_history)
        high = list(self.high_history)
        low = list(self.low_history)
        open_ = list(self.open_history)

        if method == "historical":
            vol = historical_volatility(close, 20)
        elif method == "parkinson":
            vol = parkinson_volatility(high, low, 20)
        elif method == "garman_klass":
            vol = garman_klass_volatility(open_, high, low, close, 20)
        else:
            vol = historical_volatility(close, 20)

        if len(vol) > 0 and not np.isnan(vol[-1]):
            return vol[-1]

        return 0.0

    def get_volatility_percentile(self) -> float:
        """
        Get current volatility percentile rank.

        Returns:
            Percentile from 0 to 1.
        """
        if len(self.vol_history) < 20:
            return 0.5

        vols = list(self.vol_history)
        current = vols[-1]
        below = sum(1 for v in vols if v < current)
        return below / len(vols)

    def get_volatility_regime(self) -> str:
        """
        Get current volatility regime.

        Returns:
            "low", "normal", or "high".
        """
        percentile = self.get_volatility_percentile()

        if percentile < 0.25:
            return "low"
        elif percentile > 0.75:
            return "high"
        else:
            return "normal"

    def get_volatility_trend(self, short: int = 5, long: int = 20) -> float:
        """
        Get volatility trend.

        Args:
            short: Short MA period.
            long: Long MA period.

        Returns:
            Trend value (positive = increasing, negative = decreasing).
        """
        if len(self.vol_history) < long:
            return 0.0

        vols = list(self.vol_history)
        short_ma = np.mean(vols[-short:])
        long_ma = np.mean(vols[-long:])

        if long_ma == 0:
            return 0.0

        return (short_ma - long_ma) / long_ma

    def is_high_volatility(self, threshold: float = 0.75) -> bool:
        """Check if current volatility is high."""
        return self.get_volatility_percentile() > threshold

    def is_low_volatility(self, threshold: float = 0.25) -> bool:
        """Check if current volatility is low."""
        return self.get_volatility_percentile() < threshold

    def get_volatility_forecast(self, periods: int = 5) -> float:
        """
        Simple volatility forecast using exponential smoothing.

        Args:
            periods: Forecast horizon.

        Returns:
            Forecasted volatility.
        """
        if len(self.vol_history) < 10:
            return self.get_current_volatility()

        vols = list(self.vol_history)

        # Simple exponential smoothing
        alpha = 0.3
        forecast = vols[-1]

        for _ in range(periods):
            forecast = alpha * forecast + (1 - alpha) * np.mean(vols[-10:])

        return forecast

    def reset(self) -> None:
        """Reset all history."""
        self.open_history.clear()
        self.high_history.clear()
        self.low_history.clear()
        self.close_history.clear()
        self.vol_history.clear()
