"""
Tests for database module.
"""

import pytest
from datetime import datetime, timedelta


class TestDatabase:
    """Test database operations."""

    def test_init_creates_tables(self, database):
        """Test that database initialization creates all tables."""
        stats = database.get_database_stats()

        assert "trades_count" in stats
        assert "market_snapshots_count" in stats
        assert "candles_count" in stats
        assert "strategy_stats_count" in stats
        assert "thompson_state_count" in stats

    def test_insert_and_get_trade(self, database, sample_trade_data):
        """Test inserting and retrieving a trade."""
        # Insert trade
        row_id = database.insert_trade(sample_trade_data)
        assert row_id > 0

        # Get trade
        trade = database.get_trade(sample_trade_data["trade_id"])
        assert trade is not None
        assert trade["trade_id"] == sample_trade_data["trade_id"]
        assert trade["strategy"] == sample_trade_data["strategy"]
        assert trade["direction"] == sample_trade_data["direction"]
        assert float(trade["entry_price"]) == sample_trade_data["entry_price"]

    def test_update_trade_exit(self, database, sample_trade_data, sample_exit_data):
        """Test updating trade with exit information."""
        # Insert trade
        database.insert_trade(sample_trade_data)

        # Update with exit
        success = database.update_trade_exit(
            sample_trade_data["trade_id"],
            sample_exit_data
        )
        assert success

        # Verify update
        trade = database.get_trade(sample_trade_data["trade_id"])
        assert trade["exit_price"] == sample_exit_data["exit_price"]
        assert trade["pnl"] == sample_exit_data["pnl"]
        assert trade["exit_reason"] == sample_exit_data["exit_reason"]

    def test_get_recent_trades(self, database, sample_trade_data):
        """Test getting recent trades."""
        # Insert multiple trades
        for i in range(5):
            trade = sample_trade_data.copy()
            trade["trade_id"] = f"T-{i}-test"
            database.insert_trade(trade)

        # Get recent trades
        trades = database.get_recent_trades(limit=3)
        assert len(trades) == 3

    def test_get_recent_trades_by_strategy(self, database, sample_trade_data):
        """Test filtering trades by strategy."""
        # Insert trades with different strategies
        for strategy in ["trend_follow", "mean_reversion", "trend_follow"]:
            trade = sample_trade_data.copy()
            trade["trade_id"] = f"T-{strategy}-{datetime.now().timestamp()}"
            trade["strategy"] = strategy
            database.insert_trade(trade)

        # Get trades for specific strategy
        trades = database.get_recent_trades(strategy="trend_follow")
        assert len(trades) == 2
        assert all(t["strategy"] == "trend_follow" for t in trades)

    def test_get_open_trades(self, database, sample_trade_data, sample_exit_data):
        """Test getting open trades."""
        # Insert open trade
        open_trade = sample_trade_data.copy()
        open_trade["trade_id"] = "T-open-1"
        database.insert_trade(open_trade)

        # Insert closed trade
        closed_trade = sample_trade_data.copy()
        closed_trade["trade_id"] = "T-closed-1"
        database.insert_trade(closed_trade)
        database.update_trade_exit(closed_trade["trade_id"], sample_exit_data)

        # Get open trades
        open_trades = database.get_open_trades()
        assert len(open_trades) == 1
        assert open_trades[0]["trade_id"] == "T-open-1"

    def test_features_compression(self, database, sample_trade_data):
        """Test that features are compressed and decompressed correctly."""
        features = {
            "rsi": 50.0,
            "trend": 0.5,
            "volatility": 0.02,
        }
        trade = sample_trade_data.copy()
        trade["entry_features"] = features

        database.insert_trade(trade)

        retrieved = database.get_trade(trade["trade_id"])
        assert retrieved["entry_features"] == features

    def test_insert_snapshot(self, database):
        """Test inserting market snapshots."""
        snapshot = {
            "timestamp": datetime.utcnow(),
            "bid": 50000.0,
            "ask": 50010.0,
            "mid_price": 50005.0,
            "spread_pct": 0.02,
            "imbalance_l1": 0.1,
            "imbalance_l5": 0.05,
        }

        row_id = database.insert_snapshot(snapshot)
        assert row_id > 0

    def test_insert_candle(self, database):
        """Test inserting candle data."""
        candle = {
            "timestamp": datetime.utcnow(),
            "timeframe": "5m",
            "open": 50000.0,
            "high": 50100.0,
            "low": 49900.0,
            "close": 50050.0,
            "volume": 100.0,
        }

        row_id = database.insert_candle(candle)
        assert row_id > 0

        # Test upsert (same timestamp and timeframe)
        candle["close"] = 50060.0
        database.insert_candle(candle)

        candles = database.get_candles("5m", limit=1)
        assert len(candles) == 1
        assert candles[0]["close"] == 50060.0

    def test_thompson_state(self, database):
        """Test Thompson Sampling state persistence."""
        strategies = ["trend_follow", "mean_reversion", "momentum"]

        # Initialize state
        database.init_thompson_state(strategies)

        state = database.get_thompson_state()
        assert len(state) == 3
        assert all(s["alpha"] == 1.0 for s in state.values())

        # Update state
        database.update_thompson_state("trend_follow", 5.0, 3.0, 10, 2.5)

        state = database.get_thompson_state()
        assert state["trend_follow"]["alpha"] == 5.0
        assert state["trend_follow"]["beta"] == 3.0
        assert state["trend_follow"]["total_trials"] == 10

    def test_system_state(self, database):
        """Test system state key-value store."""
        # Set state
        database.set_state("test_key", {"value": 123, "nested": {"a": 1}})

        # Get state
        value = database.get_state("test_key")
        assert value == {"value": 123, "nested": {"a": 1}}

        # Get non-existent key
        default = database.get_state("nonexistent", default="default")
        assert default == "default"

        # Delete state
        deleted = database.delete_state("test_key")
        assert deleted
        assert database.get_state("test_key") is None

    def test_strategy_stats(self, database):
        """Test strategy statistics."""
        today = datetime.utcnow().strftime("%Y-%m-%d")

        stats = {
            "trades_count": 10,
            "wins_count": 6,
            "total_pnl": 50.0,
            "win_rate": 0.6,
        }

        database.update_strategy_stats(today, "trend_follow", stats)

        retrieved = database.get_strategy_stats("trend_follow", days=1)
        assert len(retrieved) == 1
        assert retrieved[0]["trades_count"] == 10
        assert retrieved[0]["win_rate"] == 0.6

    def test_cleanup_old_data(self, database):
        """Test cleaning up old data."""
        # Insert old snapshot
        old_snapshot = {
            "timestamp": (datetime.utcnow() - timedelta(days=60)).isoformat(),
            "bid": 50000.0,
            "ask": 50010.0,
            "mid_price": 50005.0,
            "spread_pct": 0.02,
        }
        database.insert_snapshot(old_snapshot)

        # Insert recent snapshot
        recent_snapshot = old_snapshot.copy()
        recent_snapshot["timestamp"] = datetime.utcnow()
        database.insert_snapshot(recent_snapshot)

        # Cleanup
        deleted = database.cleanup_old_data(days=30)

        assert deleted["snapshots"] >= 1

    def test_export_trades_csv(self, database, sample_trade_data, sample_exit_data, tmp_path):
        """Test exporting trades to CSV."""
        # Insert and close a trade
        database.insert_trade(sample_trade_data)
        database.update_trade_exit(sample_trade_data["trade_id"], sample_exit_data)

        # Export
        csv_path = str(tmp_path / "trades.csv")
        count = database.export_trades_csv(csv_path, days=1)

        assert count >= 1

        # Verify file
        with open(csv_path) as f:
            lines = f.readlines()

        assert len(lines) >= 2  # Header + at least one trade
        assert "trade_id" in lines[0]
