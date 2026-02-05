"""
Configuration management for Paradex Trader.

Uses Pydantic for validation and supports environment variables and .env files.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class APIConfig(BaseModel):
    """Paradex API configuration."""

    l2_private_key: str = Field(..., description="Starknet L2 private key")
    l2_address: str = Field(..., description="Starknet L2 address")
    environment: str = Field(default="prod", description="API environment: prod or testnet")
    base_url: str = Field(default="", description="API base URL (auto-generated)")
    timeout: float = Field(default=10.0, ge=1.0, le=60.0, description="API timeout in seconds")
    max_retries: int = Field(default=3, ge=1, le=10, description="Max retry attempts")

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate environment value."""
        if v not in ("prod", "testnet"):
            raise ValueError("environment must be 'prod' or 'testnet'")
        return v

    def get_base_url(self) -> str:
        """Get the API base URL based on environment."""
        if self.base_url:
            return self.base_url
        if self.environment == "prod":
            return "https://api.prod.paradex.trade/v1"
        return "https://api.testnet.paradex.trade/v1"


class TradingParams(BaseModel):
    """Trading parameters configuration."""

    market: str = Field(default="BTC-USD-PERP", description="Trading market")
    base_trade_size: float = Field(
        default=0.001, gt=0, description="Base trade size in BTC"
    )
    max_position_size: float = Field(
        default=0.005, gt=0, description="Maximum position size in BTC"
    )
    min_order_size: float = Field(
        default=0.001, gt=0, description="Minimum order size in BTC"
    )
    leverage: int = Field(default=10, ge=1, le=50, description="Trading leverage")
    loop_interval_ms: int = Field(
        default=1000, ge=100, le=10000, description="Main loop interval in milliseconds"
    )

    # Entry filters
    max_spread_pct: float = Field(
        default=0.015, gt=0, le=0.1, description="Maximum spread % to enter trade"
    )
    min_signal_strength: float = Field(
        default=0.4, ge=0, le=1.0, description="Minimum signal strength to trade"
    )

    @field_validator("market")
    @classmethod
    def validate_market(cls, v: str) -> str:
        """Validate market format."""
        if not v.endswith("-PERP"):
            raise ValueError("Market must be a perpetual contract (e.g., BTC-USD-PERP)")
        return v


class RiskParams(BaseModel):
    """Risk management parameters."""

    # Per-trade limits
    risk_per_trade: float = Field(
        default=0.02, gt=0, le=0.1, description="Risk per trade (2%)"
    )
    max_position_pct: float = Field(
        default=0.10, gt=0, le=0.5, description="Max position as % of equity (10%)"
    )
    max_loss_per_trade_pct: float = Field(
        default=0.03, gt=0, le=0.1, description="Max loss per trade (3%)"
    )
    default_stop_loss_pct: float = Field(
        default=0.02, gt=0, le=0.1, description="Default stop loss (2%)"
    )
    default_take_profit_pct: float = Field(
        default=0.03, gt=0, le=0.2, description="Default take profit (3%)"
    )

    # Daily limits
    daily_loss_limit: float = Field(
        default=0.05, gt=0, le=0.2, description="Daily loss limit (5%)"
    )
    max_daily_loss_pct: float = Field(
        default=0.05, gt=0, le=0.2, description="Max daily loss (5%)"
    )
    max_daily_trades: int = Field(default=50, ge=1, description="Max trades per day")

    # Drawdown levels
    drawdown_level1: float = Field(
        default=0.03, gt=0, le=0.1, description="Level 1 drawdown threshold (3%)"
    )
    drawdown_level2: float = Field(
        default=0.05, gt=0, le=0.15, description="Level 2 drawdown threshold (5%)"
    )
    drawdown_level3: float = Field(
        default=0.10, gt=0, le=0.3, description="Level 3 drawdown threshold (10%)"
    )
    max_total_drawdown_pct: float = Field(
        default=0.15, gt=0, le=0.5, description="Max total drawdown (15%)"
    )

    # Consecutive losses and cooldown
    max_consecutive_losses: int = Field(
        default=5, ge=1, le=20, description="Max consecutive losses before cooldown"
    )
    consecutive_loss_cooldown: int = Field(
        default=3, ge=1, le=10, description="Consecutive losses to trigger cooldown"
    )
    large_loss_threshold: float = Field(
        default=0.03, gt=0, le=0.1, description="Large loss threshold (3%)"
    )
    volatility_cooldown_threshold: float = Field(
        default=0.95, gt=0, le=1.0, description="Volatility percentile to trigger cooldown"
    )
    cooldown_seconds: int = Field(
        default=300, ge=60, le=3600, description="Cooldown period after max losses (5 min)"
    )

    # Position sizing
    use_kelly_sizing: bool = Field(
        default=True, description="Use Kelly criterion for position sizing"
    )
    kelly_fraction: float = Field(
        default=0.25, gt=0, le=1.0, description="Kelly fraction (quarter-Kelly)"
    )
    min_position_multiplier: float = Field(
        default=0.5, gt=0, le=1.0, description="Minimum position size multiplier"
    )
    max_position_multiplier: float = Field(
        default=2.0, ge=1.0, le=5.0, description="Maximum position size multiplier"
    )


class StrategyConfig(BaseModel):
    """Strategy-specific configuration."""

    enabled_strategies: List[str] = Field(
        default=["trend_follow", "mean_reversion", "momentum"],
        description="List of enabled strategies",
    )

    # Trend following parameters
    trend_entry_lookback: int = Field(default=20, ge=5, le=100)
    trend_exit_lookback: int = Field(default=10, ge=3, le=50)
    trend_atr_period: int = Field(default=14, ge=5, le=50)
    trend_stop_loss_atr: float = Field(default=2.0, ge=0.5, le=5.0)
    trend_min_strength: float = Field(default=0.3, ge=0, le=1.0)

    # Mean reversion parameters
    mr_entry_z: float = Field(default=2.0, ge=1.0, le=4.0)
    mr_exit_z: float = Field(default=0.5, ge=0, le=1.5)
    mr_stop_loss_pct: float = Field(default=0.02, gt=0, le=0.1)
    mr_max_hold_minutes: int = Field(default=30, ge=5, le=120)
    mr_rsi_oversold: int = Field(default=25, ge=10, le=40)
    mr_rsi_overbought: int = Field(default=75, ge=60, le=90)

    # Momentum parameters
    mom_min_momentum_pct: float = Field(default=0.1, gt=0, le=1.0)
    mom_volume_ratio: float = Field(default=2.0, ge=1.0, le=10.0)
    mom_take_profit_pct: float = Field(default=0.0015, gt=0, le=0.01)
    mom_stop_loss_pct: float = Field(default=0.0008, gt=0, le=0.005)
    mom_max_hold_minutes: int = Field(default=3, ge=1, le=30)

    @field_validator("enabled_strategies")
    @classmethod
    def validate_strategies(cls, v: List[str]) -> List[str]:
        """Validate strategy names."""
        valid = {"trend_follow", "mean_reversion", "momentum"}
        for strategy in v:
            if strategy not in valid:
                raise ValueError(f"Invalid strategy: {strategy}. Valid: {valid}")
        return v


class LearningConfig(BaseModel):
    """Machine learning and adaptive learning configuration."""

    # Thompson Sampling
    exploration_budget: float = Field(
        default=100.0, gt=0, description="Budget for exploration phase ($)"
    )
    min_trials_per_strategy: int = Field(
        default=20, ge=5, le=100, description="Minimum trials per strategy"
    )
    min_samples_per_strategy: int = Field(
        default=20, ge=5, le=100, description="Minimum samples per strategy (alias)"
    )
    thompson_prior_alpha: float = Field(default=1.0, gt=0)
    thompson_prior_beta: float = Field(default=1.0, gt=0)
    thompson_decay: float = Field(
        default=0.995, gt=0.9, le=1.0, description="Thompson decay factor for non-stationarity"
    )
    exploitation_threshold: float = Field(
        default=0.15, ge=0, le=0.5, description="Gap required to enter exploitation mode"
    )

    # Online learning
    online_learning_enabled: bool = Field(default=True)
    online_learning_rate: float = Field(default=0.01, gt=0, le=0.5)
    online_l2_regularization: float = Field(default=0.01, ge=0, le=0.5)
    online_min_samples: int = Field(default=50, ge=5, le=200)
    drift_sensitivity: float = Field(default=0.002, gt=0, le=0.1)
    signal_threshold: float = Field(
        default=0.45, ge=0, le=1.0, description="Signal probability threshold for trading"
    )


class SystemConfig(BaseModel):
    """System and operational configuration."""

    # Data directory
    data_dir: str = Field(default="./data", description="Data storage directory")

    # Intervals
    trade_check_interval: float = Field(
        default=10.0, ge=1.0, le=60.0, description="Trade check interval (seconds)"
    )
    snapshot_interval: float = Field(
        default=5.0, ge=1.0, le=30.0, description="Market snapshot interval (seconds)"
    )
    heartbeat_interval: float = Field(
        default=60.0, ge=10.0, le=300.0, description="Heartbeat interval (seconds)"
    )

    # Database
    db_path: str = Field(default="trading.db", description="SQLite database path")

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_file: Optional[str] = Field(default="paradex_trader.log", description="Log file path")
    log_to_console: bool = Field(default=True, description="Log to console")

    # State persistence
    state_save_interval: int = Field(
        default=300, ge=60, le=3600, description="State save interval (seconds)"
    )
    state_file: str = Field(default="trader_state.json", description="State file path")

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in valid:
            raise ValueError(f"Invalid log level: {v}. Valid: {valid}")
        return v_upper


class TradingConfig(BaseSettings):
    """
    Main configuration class for Paradex Trader.

    Loads configuration from environment variables and .env file.
    All nested configurations can be overridden via environment variables
    using double underscore notation (e.g., API__L2_PRIVATE_KEY).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    # Nested configurations
    api: APIConfig = Field(default_factory=lambda: APIConfig(
        l2_private_key=os.getenv("PARADEX_L2_PRIVATE_KEY", ""),
        l2_address=os.getenv("PARADEX_L2_ADDRESS", ""),
        environment=os.getenv("PARADEX_ENVIRONMENT", "prod"),
    ))
    trading: TradingParams = Field(default_factory=TradingParams)
    risk: RiskParams = Field(default_factory=RiskParams)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    learning: LearningConfig = Field(default_factory=LearningConfig)
    system: SystemConfig = Field(default_factory=SystemConfig)

    # Shortcut environment variables (for backward compatibility)
    paradex_l2_private_key: Optional[str] = Field(default=None, alias="PARADEX_L2_PRIVATE_KEY")
    paradex_l2_address: Optional[str] = Field(default=None, alias="PARADEX_L2_ADDRESS")
    paradex_environment: Optional[str] = Field(default=None, alias="PARADEX_ENVIRONMENT")

    def model_post_init(self, __context) -> None:
        """Post-initialization processing."""
        # Apply shortcut environment variables to API config
        if self.paradex_l2_private_key:
            self.api.l2_private_key = self.paradex_l2_private_key
        if self.paradex_l2_address:
            self.api.l2_address = self.paradex_l2_address
        if self.paradex_environment:
            self.api.environment = self.paradex_environment

    def validate_required(self) -> bool:
        """
        Validate that all required configuration is present.

        Returns:
            True if valid, raises ConfigError otherwise.
        """
        from core.exceptions import ConfigError

        if not self.api.l2_private_key:
            raise ConfigError("PARADEX_L2_PRIVATE_KEY is required")
        if not self.api.l2_address:
            raise ConfigError("PARADEX_L2_ADDRESS is required")

        return True

    def get_data_dir(self) -> Path:
        """Get or create data directory."""
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        return data_dir

    def to_safe_dict(self) -> dict:
        """
        Export configuration as dict with sensitive data masked.

        Returns:
            Configuration dict with masked secrets.
        """
        config_dict = self.model_dump()

        # Mask sensitive data
        if "api" in config_dict:
            if config_dict["api"].get("l2_private_key"):
                config_dict["api"]["l2_private_key"] = "***MASKED***"

        return config_dict


@lru_cache()
def get_config() -> TradingConfig:
    """
    Get the singleton configuration instance.

    Uses LRU cache to ensure only one instance is created.

    Returns:
        TradingConfig instance.
    """
    return TradingConfig()


def reload_config() -> TradingConfig:
    """
    Reload configuration (clear cache and create new instance).

    Returns:
        New TradingConfig instance.
    """
    get_config.cache_clear()
    return get_config()


# Aliases for compatibility with main.py
Settings = TradingConfig


def load_settings(config_path: Optional[str] = None) -> TradingConfig:
    """
    Load settings from environment and optional config file.

    Args:
        config_path: Optional path to .env file.

    Returns:
        TradingConfig instance.
    """
    if config_path:
        from dotenv import load_dotenv
        load_dotenv(config_path, override=True)
        # Clear cache to reload with new env vars
        get_config.cache_clear()

    return get_config()
