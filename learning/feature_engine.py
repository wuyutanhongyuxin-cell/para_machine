"""
Feature Engineering Engine for Paradex Trader.

Extracts and maintains features across multiple timeframes:
- Tick level: Raw price movements
- 1-minute: Short-term indicators
- 5-minute: Medium-term indicators
- Orderbook: Microstructure features

Memory-efficient design using fixed-size deques.
"""

import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional

import numpy as np

logger = logging.getLogger("paradex_trader.learning.features")


@dataclass
class Candle:
    """OHLCV candle data."""
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    trades_count: int = 0

    @property
    def body(self) -> float:
        """Candle body size."""
        return abs(self.close - self.open)

    @property
    def range(self) -> float:
        """Candle range (high - low)."""
        return self.high - self.low

    @property
    def is_bullish(self) -> bool:
        """Check if candle is bullish."""
        return self.close > self.open


class FeatureEngine:
    """
    Multi-timeframe feature extraction engine.

    Maintains rolling windows of price data and computes:
    - Price change features
    - Momentum features
    - Volatility features
    - Technical indicators (RSI, Bollinger, Donchian)
    - Microstructure features (spread, imbalance)

    Memory Usage:
    - Tick history: ~50KB
    - 1min candles: ~5KB
    - 5min candles: ~5KB
    - Orderbook history: ~10KB
    Total: ~70KB (well under 100MB budget)
    """

    def __init__(
        self,
        tick_history_size: int = 500,
        candle_1m_size: int = 60,
        candle_5m_size: int = 60,
        orderbook_history_size: int = 100,
    ):
        """
        Initialize feature engine.

        Args:
            tick_history_size: Number of tick prices to keep.
            candle_1m_size: Number of 1-minute candles to keep.
            candle_5m_size: Number of 5-minute candles to keep.
            orderbook_history_size: Number of orderbook snapshots to keep.
        """
        # Tick level data
        self.tick_prices: Deque[float] = deque(maxlen=tick_history_size)
        self.tick_timestamps: Deque[float] = deque(maxlen=tick_history_size)
        self.tick_volumes: Deque[float] = deque(maxlen=tick_history_size)

        # Candle data
        self.candles_1m: Deque[Candle] = deque(maxlen=candle_1m_size)
        self.candles_5m: Deque[Candle] = deque(maxlen=candle_5m_size)

        # Current forming candles
        self._current_1m: Optional[Candle] = None
        self._current_5m: Optional[Candle] = None
        self._last_1m_minute: int = -1
        self._last_5m_slot: int = -1

        # Orderbook data
        self.imbalance_history: Deque[float] = deque(maxlen=orderbook_history_size)
        self.spread_history: Deque[float] = deque(maxlen=orderbook_history_size)

        # Current orderbook state
        self.last_imbalance: float = 0.0
        self.last_spread_pct: float = 0.0
        self.last_bid: float = 0.0
        self.last_ask: float = 0.0

        # Trade flow tracking
        self.buy_volume: Deque[float] = deque(maxlen=100)
        self.sell_volume: Deque[float] = deque(maxlen=100)

        logger.info(
            f"FeatureEngine initialized: ticks={tick_history_size}, "
            f"1m={candle_1m_size}, 5m={candle_5m_size}"
        )

    def update_price(
        self,
        price: float,
        volume: float = 0.0,
        timestamp: Optional[float] = None,
        side: Optional[str] = None,
    ) -> None:
        """
        Update with new price data.

        Args:
            price: Trade price.
            volume: Trade volume.
            timestamp: Unix timestamp (uses current time if None).
            side: Trade side ("buy" or "sell") for flow tracking.
        """
        ts = timestamp or time.time()

        self.tick_prices.append(price)
        self.tick_timestamps.append(ts)
        self.tick_volumes.append(volume)

        # Track trade flow
        if side == "buy":
            self.buy_volume.append(volume)
            self.sell_volume.append(0)
        elif side == "sell":
            self.buy_volume.append(0)
            self.sell_volume.append(volume)

        # Update candles
        self._update_candles(price, volume, ts)

    def _update_candles(self, price: float, volume: float, ts: float) -> None:
        """Update candle data."""
        minute = int(ts // 60)
        five_min = int(ts // 300)

        # 1-minute candle
        if minute != self._last_1m_minute:
            if self._current_1m is not None:
                self.candles_1m.append(self._current_1m)

            self._current_1m = Candle(
                timestamp=minute * 60,
                open=price,
                high=price,
                low=price,
                close=price,
                volume=volume,
                trades_count=1,
            )
            self._last_1m_minute = minute
        elif self._current_1m is not None:
            self._current_1m.high = max(self._current_1m.high, price)
            self._current_1m.low = min(self._current_1m.low, price)
            self._current_1m.close = price
            self._current_1m.volume += volume
            self._current_1m.trades_count += 1

        # 5-minute candle
        if five_min != self._last_5m_slot:
            if self._current_5m is not None:
                self.candles_5m.append(self._current_5m)

            self._current_5m = Candle(
                timestamp=five_min * 300,
                open=price,
                high=price,
                low=price,
                close=price,
                volume=volume,
                trades_count=1,
            )
            self._last_5m_slot = five_min
        elif self._current_5m is not None:
            self._current_5m.high = max(self._current_5m.high, price)
            self._current_5m.low = min(self._current_5m.low, price)
            self._current_5m.close = price
            self._current_5m.volume += volume
            self._current_5m.trades_count += 1

    def update_orderbook(
        self,
        bid: float,
        ask: float,
        imbalance: float,
        bid_depth: float = 0.0,
        ask_depth: float = 0.0,
    ) -> None:
        """
        Update with orderbook data.

        Args:
            bid: Best bid price.
            ask: Best ask price.
            imbalance: Orderbook imbalance (-1 to 1).
            bid_depth: Total bid depth.
            ask_depth: Total ask depth.
        """
        self.last_bid = bid
        self.last_ask = ask
        self.last_imbalance = imbalance

        mid = (bid + ask) / 2
        spread_pct = (ask - bid) / mid * 100 if mid > 0 else 0
        self.last_spread_pct = spread_pct

        self.imbalance_history.append(imbalance)
        self.spread_history.append(spread_pct)

    def get_features(self) -> Dict[str, float]:
        """
        Extract all features.

        Returns:
            Dictionary of feature names to values.
        """
        features = {}

        if len(self.tick_prices) < 10:
            return features

        prices = list(self.tick_prices)
        timestamps = list(self.tick_timestamps)
        current_price = prices[-1]

        # === Price Change Features ===
        features["price_change_30s"] = self._calc_price_change(prices, timestamps, 30)
        features["price_change_1m"] = self._calc_price_change(prices, timestamps, 60)
        features["price_change_5m"] = self._calc_price_change(prices, timestamps, 300)

        # === Momentum Features ===
        features["momentum_30s"] = self._calc_momentum(prices, timestamps, 30)
        features["momentum_1m"] = self._calc_momentum(prices, timestamps, 60)

        # === Volatility Features ===
        features["volatility_1m"] = self._calc_volatility(prices, timestamps, 60)
        features["volatility_5m"] = self._calc_volatility(prices, timestamps, 300)

        # === RSI ===
        if len(self.candles_1m) >= 14:
            closes = [c.close for c in list(self.candles_1m)[-15:]]
            features["rsi_14_1m"] = self._calc_rsi(closes)
        else:
            features["rsi_14_1m"] = 50.0

        if len(self.candles_5m) >= 14:
            closes = [c.close for c in list(self.candles_5m)[-15:]]
            features["rsi_14_5m"] = self._calc_rsi(closes)
        else:
            features["rsi_14_5m"] = 50.0

        # === ATR ===
        if len(self.candles_5m) >= 14:
            features["atr_14_5m"] = self._calc_atr(list(self.candles_5m)[-14:])
        else:
            features["atr_14_5m"] = current_price * 0.01  # Default 1%

        # === Bollinger Position ===
        if len(self.candles_5m) >= 20:
            closes = [c.close for c in list(self.candles_5m)[-20:]]
            features["bollinger_position_5m"] = self._calc_bollinger_position(
                closes, current_price
            )
        else:
            features["bollinger_position_5m"] = 0.0

        # === Trend Strength ===
        if len(self.candles_5m) >= 10:
            features["trend_strength_5m"] = self._calc_trend_strength(
                list(self.candles_5m)[-10:]
            )
        else:
            features["trend_strength_5m"] = 0.0

        # === Donchian Position ===
        if len(self.candles_5m) >= 20:
            features["donchian_position_5m"] = self._calc_donchian_position(
                list(self.candles_5m)[-20:], current_price
            )
        else:
            features["donchian_position_5m"] = 0.5

        # === Microstructure Features ===
        features["spread_pct"] = self.last_spread_pct
        features["imbalance"] = self.last_imbalance

        if len(self.spread_history) >= 20:
            spreads = list(self.spread_history)
            mean_spread = np.mean(spreads)
            std_spread = np.std(spreads)
            if std_spread > 0:
                features["spread_zscore"] = (self.last_spread_pct - mean_spread) / std_spread
            else:
                features["spread_zscore"] = 0.0

        if len(self.imbalance_history) >= 10:
            imbs = list(self.imbalance_history)
            features["imbalance_ma"] = np.mean(imbs[-10:])
            if len(imbs) >= 10:
                features["imbalance_trend"] = np.mean(imbs[-5:]) - np.mean(imbs[-10:-5])
            else:
                features["imbalance_trend"] = 0.0

        # === Volume Features ===
        if len(self.candles_1m) >= 20:
            volumes = [c.volume for c in list(self.candles_1m)[-20:]]
            if sum(volumes[:-1]) > 0:
                features["volume_ratio"] = volumes[-1] / (np.mean(volumes[:-1]) + 1e-8)
            else:
                features["volume_ratio"] = 1.0
        else:
            features["volume_ratio"] = 1.0

        # === Order Flow ===
        if len(self.buy_volume) >= 10:
            buy_sum = sum(list(self.buy_volume)[-10:])
            sell_sum = sum(list(self.sell_volume)[-10:])
            total = buy_sum + sell_sum
            if total > 0:
                features["order_flow_imbalance"] = (buy_sum - sell_sum) / total
            else:
                features["order_flow_imbalance"] = 0.0

        return features

    def _calc_price_change(
        self,
        prices: List[float],
        timestamps: List[float],
        lookback_seconds: int,
    ) -> float:
        """Calculate price change percentage over lookback period."""
        if len(prices) < 2:
            return 0.0

        now = timestamps[-1]
        target_time = now - lookback_seconds

        # Find price at target time
        old_price = prices[0]
        for i, ts in enumerate(timestamps):
            if ts <= target_time:
                old_price = prices[i]
            else:
                break

        if old_price == 0:
            return 0.0

        return (prices[-1] - old_price) / old_price * 100

    def _calc_momentum(
        self,
        prices: List[float],
        timestamps: List[float],
        lookback_seconds: int,
    ) -> float:
        """Calculate momentum using linear regression slope."""
        if len(prices) < 10:
            return 0.0

        now = timestamps[-1]

        # Get prices in lookback window
        window_prices = []
        for i, ts in enumerate(timestamps):
            if now - ts <= lookback_seconds:
                window_prices.append(prices[i])

        if len(window_prices) < 5:
            return 0.0

        # Linear regression slope
        x = np.arange(len(window_prices))
        slope = np.polyfit(x, window_prices, 1)[0]

        # Normalize to -1 to 1
        normalized = slope / (np.mean(window_prices) + 1e-8) * 100
        return max(-1.0, min(1.0, normalized))

    def _calc_volatility(
        self,
        prices: List[float],
        timestamps: List[float],
        lookback_seconds: int,
    ) -> float:
        """Calculate volatility (standard deviation of returns)."""
        now = timestamps[-1]

        window_prices = [
            prices[i] for i, ts in enumerate(timestamps)
            if now - ts <= lookback_seconds
        ]

        if len(window_prices) < 5:
            return 0.0

        returns = np.diff(window_prices) / np.array(window_prices[:-1])
        return float(np.std(returns) * 100)

    def _calc_rsi(self, closes: List[float], period: int = 14) -> float:
        """Calculate RSI."""
        if len(closes) < period + 1:
            return 50.0

        changes = np.diff(closes)
        gains = np.where(changes > 0, changes, 0)
        losses = np.where(changes < 0, -changes, 0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 100.0 if avg_gain > 0 else 50.0

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _calc_atr(self, candles: List[Candle], period: int = 14) -> float:
        """Calculate Average True Range."""
        if len(candles) < 2:
            return 0.0

        trs = []
        for i in range(1, len(candles)):
            high = candles[i].high
            low = candles[i].low
            prev_close = candles[i - 1].close

            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            trs.append(tr)

        return float(np.mean(trs[-period:])) if trs else 0.0

    def _calc_bollinger_position(
        self,
        closes: List[float],
        current_price: float,
        period: int = 20,
        num_std: float = 2.0,
    ) -> float:
        """Calculate position within Bollinger Bands (-1 to 1)."""
        if len(closes) < period:
            return 0.0

        ma = np.mean(closes[-period:])
        std = np.std(closes[-period:])

        if std == 0:
            return 0.0

        # Position: -1 = at lower band, 1 = at upper band
        position = (current_price - ma) / (num_std * std)
        return max(-1.0, min(1.0, position))

    def _calc_trend_strength(self, candles: List[Candle]) -> float:
        """Calculate trend strength (-1 to 1)."""
        if len(candles) < 5:
            return 0.0

        bullish_count = sum(1 for c in candles if c.is_bullish)
        return (bullish_count / len(candles) - 0.5) * 2

    def _calc_donchian_position(
        self,
        candles: List[Candle],
        current_price: float,
    ) -> float:
        """Calculate position within Donchian Channel (0 to 1)."""
        if len(candles) < 2:
            return 0.5

        highest = max(c.high for c in candles)
        lowest = min(c.low for c in candles)

        if highest == lowest:
            return 0.5

        return (current_price - lowest) / (highest - lowest)

    def get_donchian_breakout(self, lookback: int = 20) -> Optional[str]:
        """
        Check for Donchian channel breakout.

        Returns:
            "LONG", "SHORT", or None.
        """
        if len(self.candles_5m) < lookback + 1:
            return None

        candles = list(self.candles_5m)[-(lookback + 1):-1]
        current = self._current_5m or (
            self.candles_5m[-1] if self.candles_5m else None
        )

        if current is None:
            return None

        highest = max(c.high for c in candles)
        lowest = min(c.low for c in candles)

        if current.close > highest:
            return "LONG"
        elif current.close < lowest:
            return "SHORT"

        return None

    def get_current_price(self) -> Optional[float]:
        """Get current price."""
        return self.tick_prices[-1] if self.tick_prices else None

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            "tick_count": len(self.tick_prices),
            "candles_1m_count": len(self.candles_1m),
            "candles_5m_count": len(self.candles_5m),
            "current_price": self.get_current_price(),
            "last_spread_pct": self.last_spread_pct,
            "last_imbalance": self.last_imbalance,
        }

    def reset(self) -> None:
        """Reset all data."""
        self.tick_prices.clear()
        self.tick_timestamps.clear()
        self.tick_volumes.clear()
        self.candles_1m.clear()
        self.candles_5m.clear()
        self._current_1m = None
        self._current_5m = None
        self._last_1m_minute = -1
        self._last_5m_slot = -1
        self.imbalance_history.clear()
        self.spread_history.clear()
        self.buy_volume.clear()
        self.sell_volume.clear()

        logger.info("FeatureEngine reset")
