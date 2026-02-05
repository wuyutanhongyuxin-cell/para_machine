"""
SQLite database management for Paradex Trader.

Handles all persistent storage including:
- Trade records
- Market snapshots
- Candle data
- Strategy statistics
- Thompson Sampling state
- System state
"""

import gzip
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from core.exceptions import DatabaseError


class Database:
    """
    SQLite database manager for Paradex Trader.

    Thread-safe with connection pooling via context managers.
    All timestamps are stored as ISO format strings in UTC.
    """

    def __init__(self, db_path: str = "trading.db"):
        """
        Initialize database.

        Args:
            db_path: Path to SQLite database file.
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _init_schema(self) -> None:
        """Initialize database schema."""
        schema = """
        -- Trade records table
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_id TEXT UNIQUE NOT NULL,
            timestamp TEXT NOT NULL,
            strategy TEXT NOT NULL,
            direction TEXT NOT NULL CHECK(direction IN ('LONG', 'SHORT')),
            entry_price REAL NOT NULL,
            exit_price REAL,
            size REAL NOT NULL,
            pnl REAL,
            pnl_pct REAL,
            fees REAL DEFAULT 0,
            hold_time_seconds REAL,
            exit_reason TEXT,
            entry_features BLOB,
            market_regime TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        );

        -- Market snapshots table (for analysis and backtesting)
        CREATE TABLE IF NOT EXISTS market_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            bid REAL NOT NULL,
            ask REAL NOT NULL,
            mid_price REAL NOT NULL,
            spread_pct REAL NOT NULL,
            imbalance_l1 REAL,
            imbalance_l5 REAL,
            bid_depth REAL,
            ask_depth REAL,
            last_trade_price REAL,
            last_trade_size REAL,
            last_trade_side TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        );

        -- Candle data table
        CREATE TABLE IF NOT EXISTS candles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume REAL,
            trades_count INTEGER,
            UNIQUE(timestamp, timeframe)
        );

        -- Strategy performance statistics table
        CREATE TABLE IF NOT EXISTS strategy_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            strategy TEXT NOT NULL,
            trades_count INTEGER DEFAULT 0,
            wins_count INTEGER DEFAULT 0,
            total_pnl REAL DEFAULT 0,
            total_fees REAL DEFAULT 0,
            max_drawdown REAL DEFAULT 0,
            sharpe_ratio REAL,
            win_rate REAL,
            avg_win REAL,
            avg_loss REAL,
            profit_factor REAL,
            created_at TEXT DEFAULT (datetime('now')),
            UNIQUE(date, strategy)
        );

        -- Thompson Sampling state table
        CREATE TABLE IF NOT EXISTS thompson_state (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy TEXT UNIQUE NOT NULL,
            alpha REAL NOT NULL DEFAULT 1,
            beta REAL NOT NULL DEFAULT 1,
            total_trials INTEGER DEFAULT 0,
            total_reward REAL DEFAULT 0,
            updated_at TEXT DEFAULT (datetime('now'))
        );

        -- System state key-value store
        CREATE TABLE IF NOT EXISTS system_state (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT DEFAULT (datetime('now'))
        );

        -- Online learning model state
        CREATE TABLE IF NOT EXISTS online_model_state (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT UNIQUE NOT NULL,
            model_data BLOB NOT NULL,
            samples_seen INTEGER DEFAULT 0,
            accuracy REAL,
            precision_score REAL,
            recall REAL,
            f1_score REAL,
            drift_count INTEGER DEFAULT 0,
            updated_at TEXT DEFAULT (datetime('now'))
        );

        -- Create indexes for common queries
        CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp);
        CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy);
        CREATE INDEX IF NOT EXISTS idx_trades_trade_id ON trades(trade_id);
        CREATE INDEX IF NOT EXISTS idx_snapshots_timestamp ON market_snapshots(timestamp);
        CREATE INDEX IF NOT EXISTS idx_candles_timestamp ON candles(timestamp, timeframe);
        CREATE INDEX IF NOT EXISTS idx_strategy_stats_date ON strategy_stats(date, strategy);
        """

        with self._get_connection() as conn:
            conn.executescript(schema)

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """
        Get database connection context manager.

        Yields:
            SQLite connection with Row factory.
        """
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")  # Better concurrent access
        conn.execute("PRAGMA synchronous=NORMAL")  # Balance safety and speed

        try:
            yield conn
            conn.commit()
        except sqlite3.Error as e:
            conn.rollback()
            raise DatabaseError(f"Database error: {e}") from e
        finally:
            conn.close()

    # ==================== Trade Methods ====================

    def insert_trade(self, trade: Dict[str, Any]) -> int:
        """
        Insert a new trade record.

        Args:
            trade: Trade data dictionary.

        Returns:
            Inserted row ID.
        """
        # Compress features if present
        entry_features = trade.get("entry_features")
        if entry_features and isinstance(entry_features, dict):
            entry_features = gzip.compress(json.dumps(entry_features).encode())

        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO trades (
                    trade_id, timestamp, strategy, direction, entry_price,
                    size, entry_features, market_regime
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trade["trade_id"],
                    trade["timestamp"].isoformat() if isinstance(trade["timestamp"], datetime) else trade["timestamp"],
                    trade["strategy"],
                    trade["direction"],
                    trade["entry_price"],
                    trade["size"],
                    entry_features,
                    trade.get("market_regime"),
                ),
            )
            return cursor.lastrowid

    def update_trade_exit(self, trade_id: str, exit_data: Dict[str, Any]) -> bool:
        """
        Update trade with exit information.

        Args:
            trade_id: Trade ID to update.
            exit_data: Exit data (exit_price, pnl, pnl_pct, fees, hold_time_seconds, exit_reason).

        Returns:
            True if updated, False if trade not found.
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                UPDATE trades SET
                    exit_price = ?,
                    pnl = ?,
                    pnl_pct = ?,
                    fees = ?,
                    hold_time_seconds = ?,
                    exit_reason = ?,
                    updated_at = datetime('now')
                WHERE trade_id = ?
                """,
                (
                    exit_data["exit_price"],
                    exit_data["pnl"],
                    exit_data["pnl_pct"],
                    exit_data.get("fees", 0),
                    exit_data.get("hold_time_seconds"),
                    exit_data.get("exit_reason"),
                    trade_id,
                ),
            )
            return cursor.rowcount > 0

    def get_trade(self, trade_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a trade by ID.

        Args:
            trade_id: Trade ID.

        Returns:
            Trade data or None.
        """
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM trades WHERE trade_id = ?", (trade_id,)
            ).fetchone()

            if row:
                return self._row_to_trade_dict(row)
            return None

    def get_recent_trades(
        self,
        strategy: Optional[str] = None,
        hours: int = 24,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get recent trades.

        Args:
            strategy: Filter by strategy name.
            hours: Look back hours.
            limit: Maximum number of trades.

        Returns:
            List of trade dictionaries.
        """
        cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()

        query = "SELECT * FROM trades WHERE timestamp >= ?"
        params: List[Any] = [cutoff]

        if strategy:
            query += " AND strategy = ?"
            params.append(strategy)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_trade_dict(row) for row in rows]

    def get_open_trades(self) -> List[Dict[str, Any]]:
        """
        Get all open trades (no exit price).

        Returns:
            List of open trade dictionaries.
        """
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM trades WHERE exit_price IS NULL ORDER BY timestamp DESC"
            ).fetchall()
            return [self._row_to_trade_dict(row) for row in rows]

    def _row_to_trade_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        """Convert database row to trade dictionary."""
        trade = dict(row)

        # Decompress features if present
        if trade.get("entry_features"):
            try:
                decompressed = gzip.decompress(trade["entry_features"])
                trade["entry_features"] = json.loads(decompressed)
            except Exception:
                trade["entry_features"] = None

        return trade

    # ==================== Market Snapshot Methods ====================

    def insert_snapshot(self, snapshot: Dict[str, Any]) -> int:
        """
        Insert a market snapshot.

        Args:
            snapshot: Snapshot data.

        Returns:
            Inserted row ID.
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO market_snapshots (
                    timestamp, bid, ask, mid_price, spread_pct,
                    imbalance_l1, imbalance_l5, bid_depth, ask_depth,
                    last_trade_price, last_trade_size, last_trade_side
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot["timestamp"].isoformat() if isinstance(snapshot["timestamp"], datetime) else snapshot["timestamp"],
                    snapshot["bid"],
                    snapshot["ask"],
                    snapshot["mid_price"],
                    snapshot["spread_pct"],
                    snapshot.get("imbalance_l1"),
                    snapshot.get("imbalance_l5"),
                    snapshot.get("bid_depth"),
                    snapshot.get("ask_depth"),
                    snapshot.get("last_trade_price"),
                    snapshot.get("last_trade_size"),
                    snapshot.get("last_trade_side"),
                ),
            )
            return cursor.lastrowid

    def get_recent_snapshots(self, minutes: int = 60, limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Get recent market snapshots.

        Args:
            minutes: Look back minutes.
            limit: Maximum number of snapshots.

        Returns:
            List of snapshot dictionaries.
        """
        cutoff = (datetime.utcnow() - timedelta(minutes=minutes)).isoformat()

        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM market_snapshots
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (cutoff, limit),
            ).fetchall()
            return [dict(row) for row in rows]

    # ==================== Candle Methods ====================

    def insert_candle(self, candle: Dict[str, Any]) -> int:
        """
        Insert or update a candle.

        Args:
            candle: Candle data.

        Returns:
            Inserted/updated row ID.
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT OR REPLACE INTO candles (
                    timestamp, timeframe, open, high, low, close, volume, trades_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    candle["timestamp"].isoformat() if isinstance(candle["timestamp"], datetime) else candle["timestamp"],
                    candle["timeframe"],
                    candle["open"],
                    candle["high"],
                    candle["low"],
                    candle["close"],
                    candle.get("volume"),
                    candle.get("trades_count"),
                ),
            )
            return cursor.lastrowid

    def get_candles(
        self,
        timeframe: str,
        limit: int = 100,
        since: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get candle data.

        Args:
            timeframe: Candle timeframe (e.g., '1m', '5m', '1h').
            limit: Maximum number of candles.
            since: Get candles since this time.

        Returns:
            List of candle dictionaries.
        """
        query = "SELECT * FROM candles WHERE timeframe = ?"
        params: List[Any] = [timeframe]

        if since:
            query += " AND timestamp >= ?"
            params.append(since.isoformat())

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    # ==================== Strategy Stats Methods ====================

    def update_strategy_stats(self, date: str, strategy: str, stats: Dict[str, Any]) -> None:
        """
        Update or insert strategy statistics.

        Args:
            date: Date string (YYYY-MM-DD).
            strategy: Strategy name.
            stats: Statistics dictionary.
        """
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO strategy_stats (
                    date, strategy, trades_count, wins_count, total_pnl,
                    total_fees, max_drawdown, sharpe_ratio, win_rate,
                    avg_win, avg_loss, profit_factor
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    date,
                    strategy,
                    stats.get("trades_count", 0),
                    stats.get("wins_count", 0),
                    stats.get("total_pnl", 0),
                    stats.get("total_fees", 0),
                    stats.get("max_drawdown", 0),
                    stats.get("sharpe_ratio"),
                    stats.get("win_rate"),
                    stats.get("avg_win"),
                    stats.get("avg_loss"),
                    stats.get("profit_factor"),
                ),
            )

    def get_strategy_stats(
        self,
        strategy: str,
        days: int = 7,
    ) -> List[Dict[str, Any]]:
        """
        Get strategy statistics for recent days.

        Args:
            strategy: Strategy name.
            days: Number of days to retrieve.

        Returns:
            List of daily statistics.
        """
        cutoff = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")

        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM strategy_stats
                WHERE strategy = ? AND date >= ?
                ORDER BY date DESC
                """,
                (strategy, cutoff),
            ).fetchall()
            return [dict(row) for row in rows]

    def get_all_strategy_stats(self, days: int = 7) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get statistics for all strategies.

        Args:
            days: Number of days to retrieve.

        Returns:
            Dictionary mapping strategy names to their statistics.
        """
        cutoff = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")

        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM strategy_stats
                WHERE date >= ?
                ORDER BY strategy, date DESC
                """,
                (cutoff,),
            ).fetchall()

        result: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            strategy = row["strategy"]
            if strategy not in result:
                result[strategy] = []
            result[strategy].append(dict(row))

        return result

    # ==================== Thompson Sampling Methods ====================

    def get_thompson_state(self) -> Dict[str, Dict[str, Any]]:
        """
        Get Thompson Sampling state for all strategies.

        Returns:
            Dictionary mapping strategy names to their state.
        """
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM thompson_state").fetchall()
            return {row["strategy"]: dict(row) for row in rows}

    def update_thompson_state(
        self,
        strategy: str,
        alpha: float,
        beta: float,
        trials: int,
        total_reward: float,
    ) -> None:
        """
        Update Thompson Sampling state for a strategy.

        Args:
            strategy: Strategy name.
            alpha: Beta distribution alpha parameter.
            beta: Beta distribution beta parameter.
            trials: Total number of trials.
            total_reward: Total accumulated reward.
        """
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO thompson_state (
                    strategy, alpha, beta, total_trials, total_reward, updated_at
                ) VALUES (?, ?, ?, ?, ?, datetime('now'))
                """,
                (strategy, alpha, beta, trials, total_reward),
            )

    def init_thompson_state(self, strategies: List[str], prior_alpha: float = 1.0, prior_beta: float = 1.0) -> None:
        """
        Initialize Thompson Sampling state for strategies.

        Args:
            strategies: List of strategy names.
            prior_alpha: Prior alpha value.
            prior_beta: Prior beta value.
        """
        with self._get_connection() as conn:
            for strategy in strategies:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO thompson_state (
                        strategy, alpha, beta, total_trials, total_reward
                    ) VALUES (?, ?, ?, 0, 0)
                    """,
                    (strategy, prior_alpha, prior_beta),
                )

    # ==================== System State Methods ====================

    def get_state(self, key: str, default: Any = None) -> Any:
        """
        Get a system state value.

        Args:
            key: State key.
            default: Default value if not found.

        Returns:
            State value (JSON decoded) or default.
        """
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT value FROM system_state WHERE key = ?", (key,)
            ).fetchone()

            if row:
                try:
                    return json.loads(row["value"])
                except json.JSONDecodeError:
                    return row["value"]
            return default

    def set_state(self, key: str, value: Any) -> None:
        """
        Set a system state value.

        Args:
            key: State key.
            value: State value (will be JSON encoded).
        """
        json_value = json.dumps(value) if not isinstance(value, str) else value

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO system_state (key, value, updated_at)
                VALUES (?, ?, datetime('now'))
                """,
                (key, json_value),
            )

    def delete_state(self, key: str) -> bool:
        """
        Delete a system state value.

        Args:
            key: State key.

        Returns:
            True if deleted, False if not found.
        """
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM system_state WHERE key = ?", (key,))
            return cursor.rowcount > 0

    # ==================== Maintenance Methods ====================

    def cleanup_old_data(self, days: int = 30) -> Dict[str, int]:
        """
        Clean up old data to save space.

        Args:
            days: Delete data older than this many days.

        Returns:
            Dictionary with counts of deleted records.
        """
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()

        deleted = {}

        with self._get_connection() as conn:
            # Clean snapshots
            cursor = conn.execute(
                "DELETE FROM market_snapshots WHERE timestamp < ?", (cutoff,)
            )
            deleted["snapshots"] = cursor.rowcount

            # Clean old candles (keep more recent)
            candle_cutoff = (datetime.utcnow() - timedelta(days=days * 2)).isoformat()
            cursor = conn.execute(
                "DELETE FROM candles WHERE timestamp < ?", (candle_cutoff,)
            )
            deleted["candles"] = cursor.rowcount

            # Vacuum to reclaim space
            conn.execute("VACUUM")

        return deleted

    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.

        Returns:
            Dictionary with table counts and sizes.
        """
        stats = {}

        with self._get_connection() as conn:
            # Count records in each table
            tables = ["trades", "market_snapshots", "candles", "strategy_stats", "thompson_state", "system_state"]

            for table in tables:
                row = conn.execute(f"SELECT COUNT(*) as count FROM {table}").fetchone()
                stats[f"{table}_count"] = row["count"]

            # Get database file size
            stats["file_size_bytes"] = self.db_path.stat().st_size if self.db_path.exists() else 0
            stats["file_size_mb"] = stats["file_size_bytes"] / (1024 * 1024)

        return stats

    def export_trades_csv(self, filepath: str, days: int = 30) -> int:
        """
        Export trades to CSV file.

        Args:
            filepath: Output file path.
            days: Export trades from last N days.

        Returns:
            Number of exported trades.
        """
        import csv

        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()

        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT trade_id, timestamp, strategy, direction, entry_price,
                       exit_price, size, pnl, pnl_pct, fees, hold_time_seconds,
                       exit_reason, market_regime
                FROM trades
                WHERE timestamp >= ?
                ORDER BY timestamp
                """,
                (cutoff,),
            ).fetchall()

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "trade_id", "timestamp", "strategy", "direction", "entry_price",
                "exit_price", "size", "pnl", "pnl_pct", "fees", "hold_time_seconds",
                "exit_reason", "market_regime"
            ])
            writer.writerows([tuple(row) for row in rows])

        return len(rows)
