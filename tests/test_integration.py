"""
Integration tests for Paradex Trader.

Tests the interaction between multiple components to ensure
they work correctly together.
"""

import asyncio
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Set test environment
os.environ["PARADEX_API_KEY"] = "test_key"
os.environ["PARADEX_API_SECRET"] = "test_secret"
os.environ["PARADEX_ENVIRONMENT"] = "testnet"


class TestStrategyIntegration:
    """Test strategy components working together."""

    def test_all_strategies_generate_signals(self):
        """Test that all strategies can generate signals from context."""
        from paradex_trader.strategies.base import TradeContext, StrategyManager
        from paradex_trader.strategies import create_all_strategies
        from paradex_trader.config.settings import Settings

        settings = Settings()
        strategies = create_all_strategies(settings)
        manager = StrategyManager(strategies)

        # Create realistic context
        context = TradeContext(
            timestamp=time.time(),
            market="BTC-USD-PERP",
            mid_price=50000.0,
            bid=49995.0,
            ask=50005.0,
            spread=0.02,
            bid_size=10.0,
            ask_size=8.0,
            volume_24h=1000000.0,
            rsi=45.0,
            atr=500.0,
            atr_pct=1.0,
            bb_upper=51000.0,
            bb_lower=49000.0,
            bb_mid=50000.0,
            donchian_high=51500.0,
            donchian_low=48500.0,
            imbalance=0.1,
            ofi=0.05,
            regime="TRENDING_UP",
            regime_confidence=0.7,
            trend_strength=0.5,
            volatility_percentile=0.5,
            equity=10000.0,
            available_balance=8000.0,
        )

        signals = manager.generate_signals(context)

        # All strategies should return signals
        assert len(signals) == 3
        assert "trend_follow" in signals
        assert "mean_reversion" in signals
        assert "momentum" in signals

        # Verify signal structure
        for name, signal in signals.items():
            assert signal.direction in ["LONG", "SHORT", "HOLD"]
            assert 0 <= signal.strength <= 1
            assert signal.strategy == name


class TestLearningIntegration:
    """Test learning components working together."""

    def test_thompson_with_online_filter(self):
        """Test Thompson Sampling and online filter integration."""
        from paradex_trader.learning.thompson_sampling import ThompsonSampling
        from paradex_trader.learning.online_filter import OnlineLearningFilter
        from paradex_trader.learning.feature_engine import FeatureEngine

        strategies = ["trend_follow", "mean_reversion", "momentum"]
        thompson = ThompsonSampling(
            strategies=strategies,
            min_trials_per_strategy=5,
        )

        online_filter = OnlineLearningFilter(
            learning_rate=0.01,
            min_samples=10,
        )

        feature_engine = FeatureEngine()

        # Simulate trading loop
        for i in range(30):
            # Update features
            price = 50000 + i * 10
            volume = 100 + i
            feature_engine.update_tick(price, volume)

            features = feature_engine.get_features()

            # Get prediction (may not be ready initially)
            prediction = online_filter.predict(features)

            # Select strategy
            strategy = thompson.select_strategy()
            assert strategy in strategies

            # Simulate trade outcome
            is_win = i % 3 != 0  # 66% win rate
            pnl = 100 if is_win else -50
            pnl_pct = 1.0 if is_win else -0.5

            # Update learning
            thompson.update(strategy, pnl, pnl_pct, is_win)
            online_filter.learn(features, is_win)

        # Verify learning happened
        stats = thompson.get_stats()
        for name, s in stats.items():
            assert s["total_trials"] > 0

        filter_stats = online_filter.get_stats()
        assert filter_stats["samples_seen"] == 30

    def test_regime_detector_with_strategies(self):
        """Test regime detector integrates with strategy selection."""
        from paradex_trader.learning.regime_detector import RegimeDetector, MarketRegime

        detector = RegimeDetector(
            lookback_periods=30,
            trend_threshold=0.3,
        )

        # Simulate uptrend
        prices = [50000 + i * 50 for i in range(40)]

        for price in prices:
            state = detector.update(price)

        # Should detect trending up
        assert state.regime in [MarketRegime.TRENDING_UP, MarketRegime.BREAKOUT_UP]

        # Get strategy recommendations
        recs = detector.get_strategy_recommendations()
        assert len(recs) == 3

        # Trend following should be top recommendation in uptrend
        top_rec = recs[0]
        assert top_rec["suitability"] >= 0.5

    def test_feature_engine_produces_valid_features(self):
        """Test feature engine produces complete feature set."""
        from paradex_trader.learning.feature_engine import FeatureEngine

        engine = FeatureEngine()

        # Feed enough data for all features
        for i in range(100):
            price = 50000 + (i % 20) * 10 - 100  # Oscillating price
            volume = 100 + i % 30
            engine.update_tick(price, volume)

        features = engine.get_features()

        # Should have multiple features
        assert len(features) >= 5

        # All features should be numeric
        for name, value in features.items():
            assert isinstance(value, (int, float))
            assert not (value != value)  # Not NaN


class TestRiskIntegration:
    """Test risk management components working together."""

    def test_position_sizing_with_drawdown(self):
        """Test position sizer respects drawdown multiplier."""
        from paradex_trader.risk.position_sizer import PositionSizer
        from paradex_trader.risk.drawdown_control import DrawdownController

        sizer = PositionSizer(
            base_risk_pct=0.02,
            max_position_pct=0.10,
            kelly_fraction=0.25,
        )

        dd_controller = DrawdownController(
            initial_equity=10000.0,
            level1_threshold=0.03,
            level2_threshold=0.05,
            level3_threshold=0.10,
        )

        # Normal conditions
        size1 = sizer.calculate_size(
            equity=10000.0,
            entry_price=50000.0,
            stop_distance=500.0,
            signal_strength=0.8,
            win_rate=0.55,
        )

        # Simulate drawdown
        dd_status = dd_controller.update(9600.0)  # 4% drawdown

        # Size with drawdown reduction
        size2 = sizer.calculate_size(
            equity=9600.0,
            entry_price=50000.0,
            stop_distance=500.0,
            signal_strength=0.8,
            win_rate=0.55,
        )
        size2_adjusted = size2 * dd_status.size_multiplier

        # Adjusted size should be smaller
        assert size2_adjusted < size2
        assert dd_status.size_multiplier < 1.0

    def test_cooldown_blocks_trading(self):
        """Test cooldown manager blocks trading after losses."""
        from paradex_trader.risk.cooldown import CooldownManager

        cooldown = CooldownManager(
            consecutive_loss_threshold=3,
            large_loss_threshold=0.03,
        )

        # Simulate 3 consecutive losses
        for _ in range(3):
            cooldown.record_trade(pnl=-50, pnl_pct=-0.5, is_win=False)

        # Should be in cooldown
        assert cooldown.is_in_cooldown()

        status = cooldown.get_cooldown_status()
        assert status["in_cooldown"]
        assert "consecutive_loss" in status["reason"].lower()

    def test_complete_risk_flow(self):
        """Test complete risk management flow."""
        from paradex_trader.risk.position_sizer import PositionSizer
        from paradex_trader.risk.drawdown_control import DrawdownController
        from paradex_trader.risk.cooldown import CooldownManager

        sizer = PositionSizer(base_risk_pct=0.02)
        dd_control = DrawdownController(initial_equity=10000.0)
        cooldown = CooldownManager()

        initial_equity = 10000.0
        equity = initial_equity
        trades_blocked = 0
        trades_executed = 0

        # Simulate trading session
        for i in range(20):
            # Check drawdown
            dd_status = dd_control.update(equity)

            if dd_status.should_stop or dd_status.should_pause:
                trades_blocked += 1
                continue

            # Check cooldown
            if cooldown.is_in_cooldown():
                trades_blocked += 1
                continue

            # Calculate position size
            size = sizer.calculate_size(
                equity=equity,
                entry_price=50000.0,
                stop_distance=500.0,
                signal_strength=0.7,
            )
            size *= dd_status.size_multiplier

            if size < 0.001:
                trades_blocked += 1
                continue

            trades_executed += 1

            # Simulate random outcome
            is_win = i % 4 != 0
            pnl = 50 if is_win else -30
            equity += pnl

            cooldown.record_trade(pnl, pnl / 10000, is_win)

        # Some trades should have executed
        assert trades_executed > 0


class TestIndicatorIntegration:
    """Test indicator components working together."""

    def test_all_indicators_update(self):
        """Test all indicators can be updated together."""
        from paradex_trader.indicators.technical import TechnicalIndicators
        from paradex_trader.indicators.microstructure import MicrostructureIndicators
        from paradex_trader.indicators.volatility import VolatilityIndicators

        tech = TechnicalIndicators()
        micro = MicrostructureIndicators()
        vol = VolatilityIndicators()

        # Simulate market data stream
        for i in range(50):
            price = 50000 + (i % 10) * 100 - 500
            high = price + 50
            low = price - 50
            volume = 100 + i

            # Update technical
            tech.update({
                "open": price - 10,
                "high": high,
                "low": low,
                "close": price,
                "volume": volume,
            })

            # Update microstructure
            bids = [(price - 5 - j * 5, 10 - j) for j in range(5)]
            asks = [(price + 5 + j * 5, 10 - j) for j in range(5)]
            micro.update_orderbook(bids, asks)

            # Update volatility
            vol.update(high=high, low=low, close=price, open_price=price - 10)

        # All indicators should have values
        assert tech.get_rsi() is not None
        assert tech.get_atr() is not None

        micro_metrics = micro.get_all_metrics()
        assert "imbalance_ma" in micro_metrics

        vol_regime = vol.get_volatility_regime()
        assert vol_regime is not None


class TestDatabaseIntegration:
    """Test database integration."""

    @pytest.fixture
    def temp_db_dir(self):
        """Create temporary directory for database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.mark.asyncio
    async def test_trade_storage_and_retrieval(self, temp_db_dir):
        """Test storing and retrieving trades."""
        from paradex_trader.core.database import Database

        db = Database(temp_db_dir)
        await db.initialize()

        # Insert trade
        trade_data = {
            "trade_id": "test_trade_1",
            "timestamp": time.time(),
            "market": "BTC-USD-PERP",
            "side": "buy",
            "size": 0.1,
            "entry_price": 50000.0,
            "exit_price": 50500.0,
            "pnl": 50.0,
            "pnl_pct": 1.0,
            "strategy": "trend_follow",
            "signal_strength": 0.8,
            "entry_reason": "Breakout",
            "exit_reason": "Take profit",
            "features": {"rsi": 45, "atr": 500},
        }

        await db.insert_trade(trade_data)

        # Retrieve trades
        trades = await db.get_trades(limit=10)
        assert len(trades) == 1
        assert trades[0]["trade_id"] == "test_trade_1"

        await db.close()

    @pytest.mark.asyncio
    async def test_state_persistence(self, temp_db_dir):
        """Test system state persistence."""
        from paradex_trader.core.database import Database

        db = Database(temp_db_dir)
        await db.initialize()

        # Save state
        await db.save_system_state("peak_equity", "15000.0")
        await db.save_system_state("last_trade_time", "1234567890")

        # Retrieve state
        peak = await db.get_system_state("peak_equity")
        assert peak == "15000.0"

        last_trade = await db.get_system_state("last_trade_time")
        assert last_trade == "1234567890"

        await db.close()


class TestEndToEndSimulation:
    """End-to-end simulation tests."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary data directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_simulated_trading_session(self, temp_data_dir):
        """Simulate a complete trading session."""
        from paradex_trader.strategies.base import TradeContext, StrategyManager
        from paradex_trader.strategies import create_all_strategies
        from paradex_trader.learning.thompson_sampling import ThompsonSampling
        from paradex_trader.learning.online_filter import OnlineLearningFilter
        from paradex_trader.learning.feature_engine import FeatureEngine
        from paradex_trader.learning.regime_detector import RegimeDetector
        from paradex_trader.risk.position_sizer import PositionSizer
        from paradex_trader.risk.drawdown_control import DrawdownController
        from paradex_trader.risk.cooldown import CooldownManager
        from paradex_trader.config.settings import Settings

        # Initialize all components
        settings = Settings()
        strategies = create_all_strategies(settings)
        manager = StrategyManager(strategies)

        thompson = ThompsonSampling(
            strategies=list(strategies.keys()),
            min_trials_per_strategy=3,
        )

        online_filter = OnlineLearningFilter(min_samples=5)
        feature_engine = FeatureEngine()
        regime_detector = RegimeDetector(lookback_periods=20)

        sizer = PositionSizer(base_risk_pct=0.02)
        dd_control = DrawdownController(initial_equity=10000.0)
        cooldown = CooldownManager(consecutive_loss_threshold=5)

        # Simulation state
        equity = 10000.0
        position = None
        entry_price = 0
        entry_strategy = None
        trades_executed = 0
        total_pnl = 0

        # Generate price series (trending then ranging)
        prices = []
        # Uptrend
        for i in range(30):
            prices.append(50000 + i * 30 + (i % 5) * 10)
        # Ranging
        for i in range(30):
            prices.append(50900 + (i % 10) * 20 - 100)
        # Downtrend
        for i in range(20):
            prices.append(51000 - i * 25 - (i % 5) * 10)

        for i, price in enumerate(prices):
            # Update components
            feature_engine.update_tick(price, 100)
            regime_state = regime_detector.update(price)
            features = feature_engine.get_features()

            # Check risk conditions
            dd_status = dd_control.update(equity)
            if dd_status.should_stop:
                break

            if cooldown.is_in_cooldown():
                continue

            # Build context (simplified)
            context = TradeContext(
                timestamp=time.time(),
                market="BTC-USD-PERP",
                mid_price=price,
                bid=price - 5,
                ask=price + 5,
                spread=0.02,
                bid_size=10.0,
                ask_size=10.0,
                volume_24h=1000000.0,
                rsi=50.0,
                atr=price * 0.01,
                atr_pct=1.0,
                bb_upper=price * 1.02,
                bb_lower=price * 0.98,
                bb_mid=price,
                donchian_high=max(prices[max(0, i-20):i+1]) if i > 0 else price,
                donchian_low=min(prices[max(0, i-20):i+1]) if i > 0 else price,
                imbalance=0.0,
                ofi=0.0,
                regime=regime_state.regime.value,
                regime_confidence=regime_state.confidence,
                trend_strength=regime_state.trend_strength,
                volatility_percentile=regime_state.volatility_percentile,
                equity=equity,
                available_balance=equity * 0.8,
            )

            # Handle existing position
            if position:
                pnl_pct = (price - entry_price) / entry_price * 100
                if position == "SHORT":
                    pnl_pct = -pnl_pct

                # Exit on 1% profit or -0.5% loss
                if pnl_pct >= 1.0 or pnl_pct <= -0.5:
                    pnl = equity * (pnl_pct / 100) * 0.1  # 10% position
                    equity += pnl
                    total_pnl += pnl
                    is_win = pnl > 0

                    # Update learning
                    thompson.update(entry_strategy, pnl, pnl_pct, is_win)
                    online_filter.learn(features, is_win)
                    cooldown.record_trade(pnl, pnl_pct / 100, is_win)

                    position = None
                    trades_executed += 1
                continue

            # Generate signals
            signals = manager.generate_signals(context)
            if not signals:
                continue

            # Select strategy
            selected = thompson.select_strategy()
            if selected not in signals:
                continue

            signal = signals[selected]
            if signal.direction == "HOLD":
                continue

            # Check online filter
            prediction = online_filter.predict(features)
            if prediction.model_ready and not prediction.should_trade:
                continue

            # Enter position
            position = signal.direction
            entry_price = price
            entry_strategy = selected

        # Verify simulation completed
        assert trades_executed > 0

        # Check Thompson learning
        stats = thompson.get_stats()
        total_trials = sum(s["total_trials"] for s in stats.values())
        assert total_trials > 0

        # Save and load state
        thompson_path = Path(temp_data_dir) / "thompson.json"
        thompson.save_state(str(thompson_path))

        new_thompson = ThompsonSampling(
            strategies=list(strategies.keys()),
            min_trials_per_strategy=3,
        )
        loaded = new_thompson.load_state(str(thompson_path))
        assert loaded

        # Verify state preserved
        for name in strategies:
            assert new_thompson.arms[name].total_trials == thompson.arms[name].total_trials


class TestConfigIntegration:
    """Test configuration integration."""

    def test_settings_load_with_env(self):
        """Test settings load from environment."""
        from paradex_trader.config.settings import Settings

        # Set test environment variables
        os.environ["PARADEX_API_KEY"] = "test_key_123"
        os.environ["PARADEX_MARKET"] = "ETH-USD-PERP"
        os.environ["RISK_PER_TRADE"] = "0.015"

        settings = Settings()

        assert settings.api.api_key == "test_key_123"
        assert settings.trading.market == "ETH-USD-PERP"
        assert settings.risk.risk_per_trade == 0.015

        # Reset
        os.environ["PARADEX_MARKET"] = "BTC-USD-PERP"
        os.environ["RISK_PER_TRADE"] = "0.02"

    def test_settings_create_directories(self):
        """Test settings creates required directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["DATA_DIR"] = tmpdir

            from paradex_trader.config.settings import Settings
            settings = Settings()

            data_path = Path(settings.system.data_dir)
            assert data_path.exists() or settings.system.data_dir == tmpdir


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
