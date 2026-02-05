"""
Paradex Trader - Main Entry Point.

Self-learning cryptocurrency trading bot for Paradex DEX.
Uses Thompson Sampling for strategy selection and online learning for signal filtering.

Usage:
    python -m paradex_trader.main [--config CONFIG_PATH] [--dry-run] [--debug]
"""

import argparse
import asyncio
import logging
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from paradex_trader.config.settings import Settings, load_settings
from paradex_trader.core.client import ParadexClient
from paradex_trader.core.database import Database
from paradex_trader.core.exceptions import (
    ParadexTraderError,
    APIError,
    RiskLimitExceeded,
    InsufficientBalanceError,
)
from paradex_trader.core.models import Position, Trade, BBO
from paradex_trader.strategies.base import StrategyManager, Signal, TradeContext
from paradex_trader.strategies import create_all_strategies
from paradex_trader.learning.thompson_sampling import ThompsonSampling
from paradex_trader.learning.online_filter import OnlineLearningFilter, PredictionResult
from paradex_trader.learning.feature_engine import FeatureEngine
from paradex_trader.learning.regime_detector import RegimeDetector, MarketRegime
from paradex_trader.risk.position_sizer import PositionSizer
from paradex_trader.risk.drawdown_control import DrawdownController
from paradex_trader.risk.cooldown import CooldownManager
from paradex_trader.indicators.technical import TechnicalIndicators
from paradex_trader.indicators.microstructure import MicrostructureIndicators
from paradex_trader.indicators.volatility import VolatilityIndicators
from paradex_trader.utils.logger import setup_logger, get_logger
from paradex_trader.utils.metrics import PerformanceMetrics

logger = get_logger(__name__)


class TradingEngine:
    """
    Main trading engine that orchestrates all components.

    Responsibilities:
    - Market data collection and processing
    - Strategy signal generation
    - Thompson Sampling strategy selection
    - Online learning signal filtering
    - Risk management and position sizing
    - Order execution and management
    - State persistence
    """

    def __init__(
        self,
        settings: Settings,
        dry_run: bool = False,
    ):
        """
        Initialize trading engine.

        Args:
            settings: Configuration settings.
            dry_run: If True, simulate trades without execution.
        """
        self.settings = settings
        self.dry_run = dry_run
        self.running = False
        self._shutdown_event = asyncio.Event()

        # Core components
        self.client: Optional[ParadexClient] = None
        self.db: Optional[Database] = None

        # Strategy components
        self.strategy_manager: Optional[StrategyManager] = None
        self.thompson: Optional[ThompsonSampling] = None

        # Learning components
        self.online_filter: Optional[OnlineLearningFilter] = None
        self.feature_engine: Optional[FeatureEngine] = None
        self.regime_detector: Optional[RegimeDetector] = None

        # Risk components
        self.position_sizer: Optional[PositionSizer] = None
        self.drawdown_controller: Optional[DrawdownController] = None
        self.cooldown_manager: Optional[CooldownManager] = None

        # Indicator components
        self.tech_indicators: Optional[TechnicalIndicators] = None
        self.micro_indicators: Optional[MicrostructureIndicators] = None
        self.vol_indicators: Optional[VolatilityIndicators] = None

        # Performance tracking
        self.metrics: Optional[PerformanceMetrics] = None

        # State
        self._current_position: Optional[Position] = None
        self._pending_orders: Dict[str, Any] = {}
        self._last_trade_time: float = 0
        self._session_start: float = 0
        self._tick_count: int = 0

        logger.info(f"TradingEngine initialized (dry_run={dry_run})")

    async def initialize(self) -> None:
        """Initialize all components."""
        logger.info("=" * 60)
        logger.info("Initializing Paradex Trader...")
        logger.info("=" * 60)

        # Database (synchronous, initializes in __init__)
        db_path = Path(self.settings.system.data_dir) / "paradex_trader.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = Database(str(db_path))
        logger.info("Database initialized")

        # API Client
        self.client = ParadexClient(
            l2_private_key=self.settings.api.l2_private_key,
            l2_address=self.settings.api.l2_address,
            environment=self.settings.api.environment,
            timeout=self.settings.api.timeout,
            max_retries=self.settings.api.max_retries,
        )

        # Verify connection
        account = await self.client.get_account_info()
        logger.info(f"Connected to Paradex. Account equity: ${account.equity:,.2f}")

        # Initialize strategies
        strategies = create_all_strategies(self.settings)
        self.strategy_manager = StrategyManager(strategies)
        strategy_names = list(strategies.keys())
        logger.info(f"Strategies loaded: {strategy_names}")

        # Thompson Sampling
        self.thompson = ThompsonSampling(
            strategies=strategy_names,
            min_trials_per_strategy=self.settings.learning.min_samples_per_strategy,
            decay_factor=self.settings.learning.thompson_decay,
            use_decay=True,
        )

        # Try to load saved state
        thompson_path = Path(self.settings.system.data_dir) / "thompson_state.json"
        if thompson_path.exists():
            self.thompson.load_state(str(thompson_path))
            logger.info("Thompson Sampling state loaded")

        # Online Learning Filter
        self.online_filter = OnlineLearningFilter(
            learning_rate=self.settings.learning.online_learning_rate,
            min_samples=self.settings.learning.online_min_samples,
            trade_threshold=self.settings.learning.signal_threshold,
        )

        # Try to load saved state
        filter_path = Path(self.settings.system.data_dir) / "online_filter.pkl"
        if filter_path.exists():
            self.online_filter.load_state(str(filter_path))
            logger.info("Online filter state loaded")

        # Feature Engine
        self.feature_engine = FeatureEngine()

        # Regime Detector
        self.regime_detector = RegimeDetector(
            lookback_periods=60,
            trend_threshold=0.3,
        )

        # Risk Management
        self.position_sizer = PositionSizer(
            base_risk_pct=self.settings.risk.risk_per_trade,
            max_position_pct=self.settings.risk.max_position_pct,
            kelly_fraction=self.settings.risk.kelly_fraction,
        )

        self.drawdown_controller = DrawdownController(
            level1_threshold=self.settings.risk.drawdown_level1,
            level2_threshold=self.settings.risk.drawdown_level2,
            level3_threshold=self.settings.risk.drawdown_level3,
            daily_loss_limit=self.settings.risk.daily_loss_limit,
        )

        # Load peak from database
        peak = self.db.get_state("peak_equity")
        if peak:
            self.drawdown_controller.peak_equity = float(peak)

        self.cooldown_manager = CooldownManager(
            consecutive_loss_threshold=self.settings.risk.consecutive_loss_cooldown,
            large_loss_threshold=self.settings.risk.large_loss_threshold,
            volatility_cooldown_threshold=self.settings.risk.volatility_cooldown_threshold,
        )

        # Indicators
        self.tech_indicators = TechnicalIndicators()
        self.micro_indicators = MicrostructureIndicators()
        self.vol_indicators = VolatilityIndicators()

        # Performance Metrics
        self.metrics = PerformanceMetrics()

        logger.info("All components initialized successfully")
        logger.info("=" * 60)

    async def run(self) -> None:
        """Main trading loop."""
        self.running = True
        self._session_start = time.time()

        logger.info("Starting trading loop...")
        logger.info(f"Market: {self.settings.trading.market}")
        logger.info(f"Loop interval: {self.settings.trading.loop_interval_ms}ms")

        if self.dry_run:
            logger.warning("DRY RUN MODE - No real orders will be placed")

        try:
            while self.running and not self._shutdown_event.is_set():
                loop_start = time.time()

                try:
                    await self._trading_iteration()
                except APIError as e:
                    logger.error(f"API error in trading loop: {e}")
                    await asyncio.sleep(5)  # Wait before retry
                except RiskLimitExceeded as e:
                    logger.warning(f"Risk limit: {e}")
                except Exception as e:
                    logger.exception(f"Unexpected error in trading loop: {e}")
                    await asyncio.sleep(1)

                # Maintain loop timing
                elapsed = (time.time() - loop_start) * 1000
                sleep_time = max(0, self.settings.trading.loop_interval_ms - elapsed)

                if sleep_time > 0:
                    await asyncio.sleep(sleep_time / 1000)

                self._tick_count += 1

        except asyncio.CancelledError:
            logger.info("Trading loop cancelled")
        finally:
            await self._shutdown()

    async def _trading_iteration(self) -> None:
        """Single iteration of the trading loop."""
        market = self.settings.trading.market

        # 1. Fetch market data
        bbo = await self.client.get_bbo(market)
        orderbook = await self.client.get_orderbook(market, depth=10)

        if not bbo or bbo.bid == 0 or bbo.ask == 0:
            logger.debug("Invalid BBO data, skipping iteration")
            return

        mid_price = (bbo.bid + bbo.ask) / 2

        # 2. Update indicators
        self._update_indicators(bbo, orderbook)

        # 3. Update regime detector
        regime_state = self.regime_detector.update(mid_price)

        # 4. Update feature engine
        self.feature_engine.update_tick(mid_price, bbo.bid_size + bbo.ask_size)

        # 5. Check current position
        positions = await self.client.get_positions()
        self._current_position = positions.get(market)

        # 6. Get account info for risk calculations
        account = await self.client.get_account_info()

        # 7. Update drawdown controller
        dd_status = self.drawdown_controller.update(account.equity)

        if dd_status.should_stop:
            logger.critical(f"TRADING STOPPED: {dd_status.reason}")
            self.running = False
            return

        if dd_status.should_pause:
            logger.warning(f"Trading paused: {dd_status.reason}")
            return

        # 8. Check cooldown
        if self.cooldown_manager.is_in_cooldown():
            cooldown_info = self.cooldown_manager.get_cooldown_status()
            logger.debug(f"In cooldown: {cooldown_info}")
            return

        # 9. Build trade context
        context = self._build_trade_context(bbo, regime_state, account)

        # 10. Handle existing position
        if self._current_position and self._current_position.size != 0:
            await self._manage_position(context)
            return

        # 11. Generate signals from all strategies
        signals = self.strategy_manager.generate_signals(context)

        if not signals:
            return

        # 12. Select strategy using Thompson Sampling
        selected_strategy = self.thompson.select_strategy()

        if selected_strategy not in signals:
            return

        signal = signals[selected_strategy]

        if signal.direction == "HOLD":
            return

        # 13. Apply online learning filter
        features = self.feature_engine.get_features()
        prediction = self.online_filter.predict(features)

        if prediction.model_ready and not prediction.should_trade:
            logger.info(
                f"Signal filtered by ML: {selected_strategy} {signal.direction} "
                f"(prob={prediction.probability:.2f})"
            )
            return

        # 14. Check regime compatibility
        regime_compat = self.regime_detector.get_regime_for_strategy(selected_strategy)
        if not regime_compat["should_trade"]:
            logger.info(
                f"Signal filtered by regime: {selected_strategy} in {regime_state.regime.value}"
            )
            return

        # 15. Calculate position size
        win_rate = self.thompson.arms[selected_strategy].win_rate or 0.5
        position_size = self.position_sizer.calculate_size(
            equity=account.equity,
            entry_price=mid_price,
            stop_distance=signal.stop_loss or (mid_price * 0.01),
            signal_strength=signal.strength,
            win_rate=win_rate,
            current_volatility=self.vol_indicators.get_current_volatility() if self.vol_indicators else None,
            avg_volatility=self.vol_indicators.get_average_volatility() if self.vol_indicators else None,
        )

        # Apply drawdown reduction
        position_size *= dd_status.size_multiplier

        # Minimum size check
        min_size = self.settings.trading.min_order_size
        if position_size < min_size:
            logger.debug(f"Position size {position_size} below minimum {min_size}")
            return

        # 16. Execute trade
        await self._execute_entry(
            strategy=selected_strategy,
            signal=signal,
            size=position_size,
            context=context,
            features=features,
        )

    def _update_indicators(self, bbo: BBO, orderbook: Any) -> None:
        """Update all indicator components."""
        mid_price = (bbo.bid + bbo.ask) / 2

        # Technical indicators (need OHLCV, simplified for tick data)
        self.tech_indicators.update({
            "open": mid_price,
            "high": mid_price,
            "low": mid_price,
            "close": mid_price,
            "volume": bbo.bid_size + bbo.ask_size,
        })

        # Microstructure
        if orderbook:
            self.micro_indicators.update_orderbook(
                bids=[(l.price, l.size) for l in orderbook.bids[:5]],
                asks=[(l.price, l.size) for l in orderbook.asks[:5]],
            )

        # Volatility
        self.vol_indicators.update(
            high=mid_price,
            low=mid_price,
            close=mid_price,
            open_price=mid_price,
        )

    def _build_trade_context(
        self,
        bbo: BBO,
        regime_state: Any,
        account: Any,
    ) -> TradeContext:
        """Build trade context for strategies."""
        micro_metrics = self.micro_indicators.get_all_metrics()

        return TradeContext(
            timestamp=time.time(),
            market=self.settings.trading.market,
            mid_price=(bbo.bid + bbo.ask) / 2,
            bid=bbo.bid,
            ask=bbo.ask,
            spread=(bbo.ask - bbo.bid) / bbo.bid * 100 if bbo.bid > 0 else 0,
            bid_size=bbo.bid_size,
            ask_size=bbo.ask_size,
            volume_24h=0,  # Would need to fetch
            rsi=self.tech_indicators.get_rsi() or 50,
            atr=self.tech_indicators.get_atr() or 0,
            atr_pct=self.tech_indicators.get_atr_pct() or 0,
            bb_upper=self.tech_indicators.get_bollinger().get("upper", 0) if self.tech_indicators.get_bollinger() else 0,
            bb_lower=self.tech_indicators.get_bollinger().get("lower", 0) if self.tech_indicators.get_bollinger() else 0,
            bb_mid=self.tech_indicators.get_bollinger().get("middle", 0) if self.tech_indicators.get_bollinger() else 0,
            donchian_high=self.tech_indicators.get_donchian().get("upper", 0) if self.tech_indicators.get_donchian() else 0,
            donchian_low=self.tech_indicators.get_donchian().get("lower", 0) if self.tech_indicators.get_donchian() else 0,
            imbalance=micro_metrics.get("imbalance_ma", 0),
            ofi=micro_metrics.get("trade_flow_imbalance", 0),
            regime=regime_state.regime.value,
            regime_confidence=regime_state.confidence,
            trend_strength=regime_state.trend_strength,
            volatility_percentile=regime_state.volatility_percentile,
            equity=account.equity,
            available_balance=account.available_balance,
        )

    async def _manage_position(self, context: TradeContext) -> None:
        """Manage existing position - check exits."""
        if not self._current_position:
            return

        pos = self._current_position

        # Get exit levels from the strategy that opened the position
        # (In production, you'd store this with the trade)

        # Simple exit logic based on context
        entry_price = pos.entry_price
        current_price = context.mid_price
        pnl_pct = (current_price - entry_price) / entry_price * 100

        if pos.side == "SHORT":
            pnl_pct = -pnl_pct

        # Check stop loss (2 ATR)
        atr_pct = context.atr_pct or 1.0
        stop_pct = atr_pct * 2

        if pnl_pct <= -stop_pct:
            logger.info(f"Stop loss triggered: PnL={pnl_pct:.2f}%")
            await self._execute_exit(pos, context, "STOP_LOSS")
            return

        # Check take profit (3 ATR)
        tp_pct = atr_pct * 3
        if pnl_pct >= tp_pct:
            logger.info(f"Take profit triggered: PnL={pnl_pct:.2f}%")
            await self._execute_exit(pos, context, "TAKE_PROFIT")

    async def _execute_entry(
        self,
        strategy: str,
        signal: Signal,
        size: float,
        context: TradeContext,
        features: Dict[str, float],
    ) -> None:
        """Execute entry order."""
        market = self.settings.trading.market
        side = "BUY" if signal.direction == "LONG" else "SELL"

        logger.info(
            f"ENTRY: {strategy} {side} {size:.4f} @ {context.mid_price:.2f} "
            f"(strength={signal.strength:.2f})"
        )

        if self.dry_run:
            logger.info("DRY RUN - Order not submitted")
            # Simulate entry for learning
            self._last_trade_time = time.time()
            return

        try:
            # Place market order
            order = await self.client.place_order(
                market=market,
                side=side,
                size=size,
                order_type="MARKET",
            )

            if order:
                logger.info(f"Order placed: {order.id}")
                self._last_trade_time = time.time()

                # Store trade in database (synchronous)
                self.db.insert_trade({
                    "trade_id": order.id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "market": market,
                    "side": side.lower(),
                    "size": size,
                    "entry_price": context.mid_price,
                    "exit_price": None,
                    "pnl": None,
                    "pnl_pct": None,
                    "strategy": strategy,
                    "signal_strength": signal.strength,
                    "entry_reason": signal.reason,
                    "exit_reason": None,
                    "entry_features": features,
                })

        except Exception as e:
            logger.error(f"Order execution failed: {e}")

    async def _execute_exit(
        self,
        position: Position,
        context: TradeContext,
        reason: str,
    ) -> None:
        """Execute exit order."""
        market = self.settings.trading.market
        side = "SELL" if position.side == "LONG" else "BUY"
        size = abs(position.size)

        entry_price = position.entry_price
        exit_price = context.mid_price
        pnl = (exit_price - entry_price) * size
        if position.side == "SHORT":
            pnl = -pnl
        pnl_pct = pnl / (entry_price * size) * 100

        logger.info(
            f"EXIT: {side} {size:.4f} @ {exit_price:.2f} "
            f"(PnL=${pnl:.2f}, {pnl_pct:.2f}%, reason={reason})"
        )

        if self.dry_run:
            logger.info("DRY RUN - Order not submitted")
            return

        try:
            order = await self.client.place_order(
                market=market,
                side=side,
                size=size,
                order_type="MARKET",
            )

            if order:
                is_win = pnl > 0

                # Update Thompson Sampling
                # (Need to track which strategy opened the position)
                # For now, update all strategies proportionally
                for name, arm in self.thompson.arms.items():
                    if arm.last_selected > 0:
                        self.thompson.update(name, pnl, pnl_pct, is_win)
                        break

                # Update online filter
                features = self.feature_engine.get_features()
                self.online_filter.learn(features, is_win)

                # Update cooldown
                self.cooldown_manager.record_trade(
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    is_win=is_win,
                )

                # Update metrics
                self.metrics.add_trade(pnl, pnl_pct)

                logger.info(f"Exit order placed: {order.id}")

        except Exception as e:
            logger.error(f"Exit execution failed: {e}")

    async def _shutdown(self) -> None:
        """Shutdown and save state."""
        logger.info("Shutting down trading engine...")

        # Save Thompson Sampling state
        if self.thompson:
            thompson_path = Path(self.settings.system.data_dir) / "thompson_state.json"
            self.thompson.save_state(str(thompson_path))
            logger.info("Thompson Sampling state saved")

        # Save online filter state
        if self.online_filter:
            filter_path = Path(self.settings.system.data_dir) / "online_filter.pkl"
            self.online_filter.save_state(str(filter_path))
            logger.info("Online filter state saved")

        # Save peak equity (synchronous)
        if self.drawdown_controller and self.db:
            self.db.set_state(
                "peak_equity",
                str(self.drawdown_controller.peak_equity)
            )

        # Database doesn't need explicit close (uses context managers)

        # Close API client
        if self.client:
            await self.client.close()

        # Print session summary
        self._print_session_summary()

        logger.info("Shutdown complete")

    def _print_session_summary(self) -> None:
        """Print trading session summary."""
        duration = time.time() - self._session_start
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)

        logger.info("=" * 60)
        logger.info("SESSION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Duration: {hours}h {minutes}m")
        logger.info(f"Total ticks: {self._tick_count}")

        if self.thompson:
            summary = self.thompson.get_summary()
            logger.info(f"Best strategy: {summary['best_strategy']}")
            logger.info(f"Exploration mode: {summary['exploration_mode']}")

            for name, stats in summary["strategies"].items():
                logger.info(
                    f"  {name}: trials={stats['trials']}, "
                    f"win_rate={stats['win_rate']:.1%}, "
                    f"PnL=${stats['pnl']:.2f}"
                )

        if self.metrics:
            metrics = self.metrics.get_all()
            logger.info(f"Total PnL: ${metrics.get('total_pnl', 0):.2f}")
            logger.info(f"Win rate: {metrics.get('win_rate', 0):.1%}")
            logger.info(f"Sharpe: {metrics.get('sharpe_ratio', 0):.2f}")

        if self.drawdown_controller:
            dd_stats = self.drawdown_controller.get_stats()
            logger.info(f"Max drawdown: {dd_stats['max_drawdown_pct']:.2f}%")

        logger.info("=" * 60)

    def stop(self) -> None:
        """Request graceful shutdown."""
        logger.info("Shutdown requested")
        self.running = False
        self._shutdown_event.set()


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Paradex Trader - Self-learning crypto trading bot"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file (.env)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in simulation mode without placing real orders",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logger(level=log_level)

    # Load settings
    try:
        if args.config:
            settings = load_settings(args.config)
        else:
            settings = load_settings()
    except Exception as e:
        logger.error(f"Failed to load settings: {e}")
        sys.exit(1)

    # Create and run engine
    engine = TradingEngine(settings, dry_run=args.dry_run)

    # Setup signal handlers
    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}")
        engine.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        await engine.initialize()
        await engine.run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        engine.stop()
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
