"""
Pytest configuration and fixtures for Paradex Trader tests.
"""

import asyncio
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Generator
from unittest.mock import AsyncMock, MagicMock

import pytest

# Set test environment before importing modules
os.environ["PARADEX_L2_PRIVATE_KEY"] = "test_private_key"
os.environ["PARADEX_L2_ADDRESS"] = "test_address"
os.environ["PARADEX_ENVIRONMENT"] = "testnet"


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_db_path(tmp_path: Path) -> str:
    """Provide a temporary database path."""
    return str(tmp_path / "test_trading.db")


@pytest.fixture
def database(temp_db_path: str):
    """Provide a test database instance."""
    from core.database import Database

    db = Database(temp_db_path)
    yield db

    # Cleanup
    if Path(temp_db_path).exists():
        Path(temp_db_path).unlink()


@pytest.fixture
def mock_client():
    """Provide a mock Paradex client."""
    client = AsyncMock()

    # Configure default return values
    client.get_bbo.return_value = MagicMock(
        bid=50000.0,
        ask=50010.0,
        bid_size=1.0,
        ask_size=1.0,
        mid_price=50005.0,
        spread=10.0,
        spread_pct=0.02,
    )

    client.get_balance.return_value = 1000.0
    client.get_positions.return_value = []
    client.has_open_position.return_value = False
    client.ping.return_value = True

    return client


@pytest.fixture
def sample_trade_data():
    """Provide sample trade data."""
    return {
        "trade_id": "T-1234567890-abc123",
        "timestamp": datetime.utcnow(),
        "strategy": "trend_follow",
        "direction": "LONG",
        "entry_price": 50000.0,
        "size": 0.001,
        "entry_features": {
            "rsi_14_5m": 45.0,
            "trend_strength_5m": 0.5,
            "spread_pct": 0.02,
        },
        "market_regime": "TRENDING_UP",
    }


@pytest.fixture
def sample_exit_data():
    """Provide sample exit data."""
    return {
        "exit_price": 50500.0,
        "pnl": 0.5,
        "pnl_pct": 0.01,
        "fees": 0.0,
        "hold_time_seconds": 300.0,
        "exit_reason": "take_profit",
    }


@pytest.fixture
def sample_features():
    """Provide sample feature data."""
    return {
        "price_change_30s": 0.05,
        "price_change_1m": 0.08,
        "price_change_5m": 0.15,
        "momentum_30s": 0.3,
        "momentum_1m": 0.25,
        "volatility_1m": 0.02,
        "volatility_5m": 0.03,
        "rsi_14_1m": 55.0,
        "rsi_14_5m": 52.0,
        "bollinger_position_5m": 0.3,
        "trend_strength_5m": 0.4,
        "donchian_position_5m": 0.7,
        "spread_pct": 0.015,
        "imbalance": 0.2,
        "volume_ratio": 1.5,
    }


@pytest.fixture
def sample_candles():
    """Provide sample candle data."""
    import time

    candles = []
    base_time = time.time() - 3600  # 1 hour ago
    base_price = 50000.0

    for i in range(60):
        price_change = (i % 10 - 5) * 10  # Oscillate
        candles.append({
            "timestamp": datetime.fromtimestamp(base_time + i * 60),
            "timeframe": "1m",
            "open": base_price + price_change,
            "high": base_price + price_change + 20,
            "low": base_price + price_change - 20,
            "close": base_price + price_change + 10,
            "volume": 100 + i * 5,
        })

    return candles


@pytest.fixture
def trading_config():
    """Provide a test trading configuration."""
    from config.settings import TradingConfig

    return TradingConfig()


class MockResponse:
    """Mock aiohttp response."""

    def __init__(self, data: dict, status: int = 200):
        self._data = data
        self.status = status
        self.headers = {}

    async def json(self):
        return self._data

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


@pytest.fixture
def mock_response():
    """Provide mock response factory."""
    return MockResponse
