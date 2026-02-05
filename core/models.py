"""Data models for Paradex Trader."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Optional, Any


class OrderSide(Enum):
    """Order side."""

    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    """Order type."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"


class OrderStatus(Enum):
    """Order status."""

    PENDING = "PENDING"
    OPEN = "OPEN"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class PositionSide(Enum):
    """Position side."""

    LONG = "LONG"
    SHORT = "SHORT"


class MarketRegime(Enum):
    """Market regime classification."""

    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING = "RANGING"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    UNKNOWN = "UNKNOWN"


@dataclass
class BBO:
    """Best Bid and Offer."""

    bid: float
    ask: float
    bid_size: float
    ask_size: float
    timestamp: float

    @property
    def mid_price(self) -> float:
        """Calculate mid price."""
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> float:
        """Calculate absolute spread."""
        return self.ask - self.bid

    @property
    def spread_pct(self) -> float:
        """Calculate spread as percentage."""
        if self.mid_price == 0:
            return 0.0
        return (self.spread / self.mid_price) * 100


@dataclass
class OrderbookLevel:
    """Single orderbook level."""

    price: float
    size: float


@dataclass
class Orderbook:
    """Orderbook snapshot."""

    bids: list[OrderbookLevel]
    asks: list[OrderbookLevel]
    timestamp: float

    @property
    def best_bid(self) -> Optional[float]:
        """Get best bid price."""
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        """Get best ask price."""
        return self.asks[0].price if self.asks else None

    @property
    def mid_price(self) -> Optional[float]:
        """Calculate mid price."""
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return None

    def calculate_imbalance(self, levels: int = 5) -> float:
        """
        Calculate order book imbalance.

        Returns:
            Imbalance between -1 (sell pressure) and 1 (buy pressure).
        """
        bid_volume = sum(level.size for level in self.bids[:levels])
        ask_volume = sum(level.size for level in self.asks[:levels])
        total = bid_volume + ask_volume

        if total == 0:
            return 0.0

        return (bid_volume - ask_volume) / total


@dataclass
class Trade:
    """Trade record."""

    trade_id: str
    timestamp: datetime
    strategy: str
    direction: str  # "LONG" or "SHORT"
    entry_price: float
    size: float
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    fees: float = 0.0
    hold_time_seconds: Optional[float] = None
    exit_reason: Optional[str] = None
    entry_features: Optional[Dict[str, float]] = None
    market_regime: Optional[str] = None

    @property
    def is_closed(self) -> bool:
        """Check if trade is closed."""
        return self.exit_price is not None

    @property
    def is_profitable(self) -> bool:
        """Check if trade is profitable."""
        if self.pnl is None:
            return False
        return self.pnl > 0


@dataclass
class Position:
    """Current position."""

    market: str
    side: PositionSide
    size: float
    entry_price: float
    unrealized_pnl: float
    leverage: int
    liquidation_price: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Order:
    """Order information."""

    order_id: str
    market: str
    side: OrderSide
    order_type: OrderType
    size: float
    price: Optional[float]
    status: OrderStatus
    filled_size: float = 0.0
    avg_fill_price: Optional[float] = None
    fees: float = 0.0
    client_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class AccountInfo:
    """Account information."""

    balance: float
    available_balance: float
    unrealized_pnl: float
    margin_used: float
    leverage: int
    positions: list[Position] = field(default_factory=list)


@dataclass
class MarketInfo:
    """Market information."""

    symbol: str
    base_currency: str
    quote_currency: str
    min_size: float
    size_increment: float
    price_increment: float
    max_leverage: int


@dataclass
class StrategyStats:
    """Strategy performance statistics."""

    strategy: str
    trades_count: int
    wins_count: int
    total_pnl: float
    total_fees: float
    max_drawdown: float
    sharpe_ratio: Optional[float]
    win_rate: Optional[float]
    avg_win: Optional[float]
    avg_loss: Optional[float]
    profit_factor: Optional[float]
    period_start: datetime
    period_end: datetime

    @property
    def losses_count(self) -> int:
        """Get number of losing trades."""
        return self.trades_count - self.wins_count

    @property
    def net_pnl(self) -> float:
        """Get net PnL after fees."""
        return self.total_pnl - self.total_fees
