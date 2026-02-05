"""
Tests for configuration module.
"""

import os
import pytest


class TestTradingConfig:
    """Test trading configuration."""

    def test_default_values(self, trading_config):
        """Test default configuration values."""
        assert trading_config.trading.market == "BTC-USD-PERP"
        assert trading_config.trading.base_trade_size == 0.001
        assert trading_config.trading.leverage == 10
        assert trading_config.risk.max_consecutive_losses == 5

    def test_strategy_config(self, trading_config):
        """Test strategy configuration."""
        assert "trend_follow" in trading_config.strategy.enabled_strategies
        assert "mean_reversion" in trading_config.strategy.enabled_strategies
        assert "momentum" in trading_config.strategy.enabled_strategies

    def test_risk_params(self, trading_config):
        """Test risk parameters."""
        assert trading_config.risk.max_daily_loss_pct == 0.05
        assert trading_config.risk.max_total_drawdown_pct == 0.15
        assert trading_config.risk.cooldown_seconds == 300

    def test_learning_config(self, trading_config):
        """Test learning configuration."""
        assert trading_config.learning.min_trials_per_strategy == 20
        assert trading_config.learning.online_learning_enabled is True
        assert trading_config.learning.drift_sensitivity == 0.002

    def test_system_config(self, trading_config):
        """Test system configuration."""
        assert trading_config.system.trade_check_interval == 10.0
        assert trading_config.system.snapshot_interval == 5.0
        assert trading_config.system.log_level == "INFO"

    def test_api_config(self, trading_config):
        """Test API configuration."""
        assert trading_config.api.environment in ("prod", "testnet")
        assert trading_config.api.timeout == 10.0
        assert trading_config.api.max_retries == 3

    def test_get_base_url(self, trading_config):
        """Test API base URL generation."""
        trading_config.api.environment = "prod"
        assert "prod" in trading_config.api.get_base_url()

        trading_config.api.environment = "testnet"
        assert "testnet" in trading_config.api.get_base_url()

    def test_to_safe_dict(self, trading_config):
        """Test safe dictionary export (masks secrets)."""
        trading_config.api.l2_private_key = "secret_key"

        safe_dict = trading_config.to_safe_dict()

        assert safe_dict["api"]["l2_private_key"] == "***MASKED***"


class TestAPIConfig:
    """Test API configuration."""

    def test_validate_environment(self):
        """Test environment validation."""
        from config.settings import APIConfig

        # Valid environments
        config = APIConfig(
            l2_private_key="test",
            l2_address="test",
            environment="prod"
        )
        assert config.environment == "prod"

        config = APIConfig(
            l2_private_key="test",
            l2_address="test",
            environment="testnet"
        )
        assert config.environment == "testnet"

        # Invalid environment
        with pytest.raises(ValueError):
            APIConfig(
                l2_private_key="test",
                l2_address="test",
                environment="invalid"
            )


class TestTradingParams:
    """Test trading parameters."""

    def test_validate_market(self):
        """Test market symbol validation."""
        from config.settings import TradingParams

        # Valid market
        params = TradingParams(market="ETH-USD-PERP")
        assert params.market == "ETH-USD-PERP"

        # Invalid market
        with pytest.raises(ValueError):
            TradingParams(market="BTC-USD")  # Not a PERP


class TestStrategyConfig:
    """Test strategy configuration."""

    def test_validate_strategies(self):
        """Test strategy validation."""
        from config.settings import StrategyConfig

        # Valid strategies
        config = StrategyConfig(enabled_strategies=["trend_follow", "momentum"])
        assert len(config.enabled_strategies) == 2

        # Invalid strategy
        with pytest.raises(ValueError):
            StrategyConfig(enabled_strategies=["invalid_strategy"])
