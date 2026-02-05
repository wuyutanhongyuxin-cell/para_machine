"""
Microstructure Indicators for Paradex Trader.

Orderbook-based and trade-based indicators for market microstructure analysis.
"""

import numpy as np
from collections import deque
from typing import List, Optional, Tuple, Deque


class MicrostructureIndicators:
    """
    Market microstructure indicators.

    Calculates:
    - Order book imbalance
    - Order flow imbalance (OFI)
    - Bid-ask spread metrics
    - Volume imbalance
    - Trade flow analysis
    """

    def __init__(self, history_size: int = 100):
        """
        Initialize microstructure indicators.

        Args:
            history_size: Number of historical values to keep.
        """
        self.history_size = history_size

        # Orderbook history
        self.imbalance_history: Deque[float] = deque(maxlen=history_size)
        self.spread_history: Deque[float] = deque(maxlen=history_size)

        # Trade flow
        self.buy_volume: Deque[float] = deque(maxlen=history_size)
        self.sell_volume: Deque[float] = deque(maxlen=history_size)
        self.trade_prices: Deque[float] = deque(maxlen=history_size)

        # OFI tracking
        self._prev_bid: float = 0
        self._prev_ask: float = 0
        self._prev_bid_size: float = 0
        self._prev_ask_size: float = 0
        self.ofi_history: Deque[float] = deque(maxlen=history_size)

    def update_orderbook(
        self,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
    ) -> dict:
        """
        Update with new orderbook snapshot.

        Args:
            bids: List of (price, size) tuples for bids.
            asks: List of (price, size) tuples for asks.

        Returns:
            Dictionary of calculated metrics.
        """
        if not bids or not asks:
            return {}

        best_bid = bids[0][0]
        best_ask = asks[0][0]
        best_bid_size = bids[0][1]
        best_ask_size = asks[0][1]

        # Calculate metrics
        metrics = {}

        # Spread
        spread = best_ask - best_bid
        mid_price = (best_bid + best_ask) / 2
        spread_pct = (spread / mid_price) * 100 if mid_price > 0 else 0

        metrics["spread"] = spread
        metrics["spread_pct"] = spread_pct
        metrics["mid_price"] = mid_price
        self.spread_history.append(spread_pct)

        # Level 1 imbalance
        imbalance_l1 = self._calc_imbalance(best_bid_size, best_ask_size)
        metrics["imbalance_l1"] = imbalance_l1

        # Multi-level imbalance
        levels = min(5, len(bids), len(asks))
        bid_depth = sum(bids[i][1] for i in range(levels))
        ask_depth = sum(asks[i][1] for i in range(levels))
        imbalance_l5 = self._calc_imbalance(bid_depth, ask_depth)
        metrics["imbalance_l5"] = imbalance_l5
        metrics["bid_depth"] = bid_depth
        metrics["ask_depth"] = ask_depth

        self.imbalance_history.append(imbalance_l5)

        # Order Flow Imbalance (OFI)
        ofi = self._calc_ofi(best_bid, best_ask, best_bid_size, best_ask_size)
        metrics["ofi"] = ofi
        self.ofi_history.append(ofi)

        # Update previous values
        self._prev_bid = best_bid
        self._prev_ask = best_ask
        self._prev_bid_size = best_bid_size
        self._prev_ask_size = best_ask_size

        # Derived metrics
        metrics["imbalance_ma"] = self.get_imbalance_ma()
        metrics["spread_percentile"] = self.get_spread_percentile()

        return metrics

    def update_trade(
        self,
        price: float,
        size: float,
        side: str,
    ) -> dict:
        """
        Update with new trade.

        Args:
            price: Trade price.
            size: Trade size.
            side: Trade side ("buy" or "sell").

        Returns:
            Dictionary of trade flow metrics.
        """
        self.trade_prices.append(price)

        if side.lower() == "buy":
            self.buy_volume.append(size)
            self.sell_volume.append(0)
        else:
            self.buy_volume.append(0)
            self.sell_volume.append(size)

        return {
            "trade_flow_imbalance": self.get_trade_flow_imbalance(),
            "buy_pressure": self.get_buy_pressure(),
        }

    def _calc_imbalance(self, bid_size: float, ask_size: float) -> float:
        """
        Calculate imbalance between bid and ask.

        Returns value from -1 (all asks) to 1 (all bids).
        """
        total = bid_size + ask_size
        if total == 0:
            return 0.0
        return (bid_size - ask_size) / total

    def _calc_ofi(
        self,
        bid: float,
        ask: float,
        bid_size: float,
        ask_size: float,
    ) -> float:
        """
        Calculate Order Flow Imbalance (OFI).

        OFI captures changes in orderbook pressure.
        """
        if self._prev_bid == 0:
            return 0.0

        ofi = 0.0

        # Bid side contribution
        if bid >= self._prev_bid:
            ofi += bid_size
        else:
            ofi -= self._prev_bid_size

        # Ask side contribution
        if ask <= self._prev_ask:
            ofi -= ask_size
        else:
            ofi += self._prev_ask_size

        # Normalize
        return ofi / 1000  # Scale down

    def get_imbalance_ma(self, period: int = 10) -> float:
        """Get moving average of imbalance."""
        if len(self.imbalance_history) < period:
            return 0.0
        return np.mean(list(self.imbalance_history)[-period:])

    def get_imbalance_trend(self, short: int = 5, long: int = 20) -> float:
        """Get imbalance trend (short MA - long MA)."""
        if len(self.imbalance_history) < long:
            return 0.0

        hist = list(self.imbalance_history)
        short_ma = np.mean(hist[-short:])
        long_ma = np.mean(hist[-long:])
        return short_ma - long_ma

    def get_spread_percentile(self) -> float:
        """Get current spread percentile rank."""
        if len(self.spread_history) < 10:
            return 0.5

        spreads = list(self.spread_history)
        current = spreads[-1]
        below = sum(1 for s in spreads if s < current)
        return below / len(spreads)

    def get_spread_zscore(self) -> float:
        """Get current spread Z-score."""
        if len(self.spread_history) < 20:
            return 0.0

        spreads = list(self.spread_history)
        mean = np.mean(spreads)
        std = np.std(spreads)

        if std == 0:
            return 0.0

        return (spreads[-1] - mean) / std

    def get_trade_flow_imbalance(self, period: int = 10) -> float:
        """Get trade flow imbalance over recent trades."""
        if len(self.buy_volume) < period:
            return 0.0

        buy = sum(list(self.buy_volume)[-period:])
        sell = sum(list(self.sell_volume)[-period:])
        total = buy + sell

        if total == 0:
            return 0.0

        return (buy - sell) / total

    def get_buy_pressure(self, period: int = 20) -> float:
        """Get buy pressure (buy volume / total volume)."""
        if len(self.buy_volume) < period:
            return 0.5

        buy = sum(list(self.buy_volume)[-period:])
        sell = sum(list(self.sell_volume)[-period:])
        total = buy + sell

        if total == 0:
            return 0.5

        return buy / total

    def get_ofi_signal(self, threshold: float = 0.5) -> Optional[str]:
        """
        Get trading signal from OFI.

        Args:
            threshold: OFI threshold for signal.

        Returns:
            "LONG", "SHORT", or None.
        """
        if len(self.ofi_history) < 5:
            return None

        ofi_ma = np.mean(list(self.ofi_history)[-5:])

        if ofi_ma > threshold:
            return "LONG"
        elif ofi_ma < -threshold:
            return "SHORT"

        return None

    def get_all_metrics(self) -> dict:
        """Get all current metrics."""
        return {
            "imbalance_ma": self.get_imbalance_ma(),
            "imbalance_trend": self.get_imbalance_trend(),
            "spread_percentile": self.get_spread_percentile(),
            "spread_zscore": self.get_spread_zscore(),
            "trade_flow_imbalance": self.get_trade_flow_imbalance(),
            "buy_pressure": self.get_buy_pressure(),
            "ofi_signal": self.get_ofi_signal(),
            "data_points": len(self.imbalance_history),
        }

    def reset(self) -> None:
        """Reset all history."""
        self.imbalance_history.clear()
        self.spread_history.clear()
        self.buy_volume.clear()
        self.sell_volume.clear()
        self.trade_prices.clear()
        self.ofi_history.clear()
        self._prev_bid = 0
        self._prev_ask = 0
        self._prev_bid_size = 0
        self._prev_ask_size = 0
