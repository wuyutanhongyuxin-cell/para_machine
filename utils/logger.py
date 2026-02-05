"""
Logging configuration for Paradex Trader.

Provides structured logging with both console and file output.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Global logger cache
_loggers: dict[str, logging.Logger] = {}


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color support for console output."""

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def __init__(self, fmt: str = None, datefmt: str = None, use_colors: bool = True):
        super().__init__(fmt, datefmt)
        self.use_colors = use_colors

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        # Save original levelname
        original_levelname = record.levelname

        if self.use_colors and record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.RESET}"

        result = super().format(record)

        # Restore original levelname
        record.levelname = original_levelname

        return result


class TradeFormatter(logging.Formatter):
    """Specialized formatter for trade-related logs."""

    def format(self, record: logging.LogRecord) -> str:
        """Format trade log record."""
        # Add trade-specific formatting if available
        if hasattr(record, "trade_id"):
            record.msg = f"[{record.trade_id}] {record.msg}"
        if hasattr(record, "strategy"):
            record.msg = f"[{record.strategy}] {record.msg}"

        return super().format(record)


def setup_logger(
    name: str = "paradex_trader",
    level: str = "INFO",
    log_file: Optional[str] = None,
    log_to_console: bool = True,
    log_dir: str = "logs",
) -> logging.Logger:
    """
    Set up a logger with configured handlers.

    Args:
        name: Logger name.
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Log file name (if None, creates timestamped file).
        log_to_console: Whether to output to console.
        log_dir: Directory for log files.

    Returns:
        Configured logger instance.
    """
    # Return cached logger if exists
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers.clear()  # Remove any existing handlers

    # Console format with colors
    console_format = "%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s"
    console_datefmt = "%H:%M:%S"

    # File format (more detailed)
    file_format = "%(asctime)s │ %(levelname)-8s │ %(name)s │ %(filename)s:%(lineno)d │ %(message)s"
    file_datefmt = "%Y-%m-%d %H:%M:%S"

    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            ColoredFormatter(fmt=console_format, datefmt=console_datefmt)
        )
        logger.addHandler(console_handler)

    # File handler
    if log_file is not None or log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(exist_ok=True)

        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d")
            log_file = f"paradex_trader_{timestamp}.log"

        file_handler = logging.FileHandler(
            log_path / log_file, encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter(fmt=file_format, datefmt=file_datefmt)
        )
        logger.addHandler(file_handler)

    # Cache logger
    _loggers[name] = logger

    return logger


def get_logger(name: str = "paradex_trader") -> logging.Logger:
    """
    Get a logger instance.

    If the logger doesn't exist, creates one with default settings.

    Args:
        name: Logger name.

    Returns:
        Logger instance.
    """
    if name in _loggers:
        return _loggers[name]

    return setup_logger(name)


def get_trade_logger() -> logging.Logger:
    """
    Get a logger specifically for trade operations.

    Returns:
        Trade-specific logger.
    """
    return get_logger("paradex_trader.trades")


def get_strategy_logger(strategy_name: str) -> logging.Logger:
    """
    Get a logger for a specific strategy.

    Args:
        strategy_name: Name of the strategy.

    Returns:
        Strategy-specific logger.
    """
    return get_logger(f"paradex_trader.strategy.{strategy_name}")


class LogContext:
    """Context manager for adding extra fields to log records."""

    def __init__(self, logger: logging.Logger, **extra):
        self.logger = logger
        self.extra = extra
        self.old_factory = None

    def __enter__(self):
        self.old_factory = logging.getLogRecordFactory()

        extra = self.extra

        def record_factory(*args, **kwargs):
            record = self.old_factory(*args, **kwargs)
            for key, value in extra.items():
                setattr(record, key, value)
            return record

        logging.setLogRecordFactory(record_factory)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.setLogRecordFactory(self.old_factory)
        return False


def log_trade_entry(
    logger: logging.Logger,
    trade_id: str,
    strategy: str,
    direction: str,
    price: float,
    size: float,
    **extra,
) -> None:
    """
    Log a trade entry.

    Args:
        logger: Logger instance.
        trade_id: Trade ID.
        strategy: Strategy name.
        direction: LONG or SHORT.
        price: Entry price.
        size: Position size.
        **extra: Additional fields to log.
    """
    msg = f"ENTRY │ {direction} │ Price: {price:.2f} │ Size: {size}"
    if extra:
        extra_str = " │ ".join(f"{k}: {v}" for k, v in extra.items())
        msg = f"{msg} │ {extra_str}"

    with LogContext(logger, trade_id=trade_id, strategy=strategy):
        logger.info(msg)


def log_trade_exit(
    logger: logging.Logger,
    trade_id: str,
    strategy: str,
    direction: str,
    entry_price: float,
    exit_price: float,
    pnl: float,
    pnl_pct: float,
    reason: str,
    hold_time: float,
) -> None:
    """
    Log a trade exit.

    Args:
        logger: Logger instance.
        trade_id: Trade ID.
        strategy: Strategy name.
        direction: LONG or SHORT.
        entry_price: Entry price.
        exit_price: Exit price.
        pnl: Profit/loss amount.
        pnl_pct: Profit/loss percentage.
        reason: Exit reason.
        hold_time: Hold time in seconds.
    """
    pnl_symbol = "+" if pnl >= 0 else ""
    result = "WIN" if pnl >= 0 else "LOSS"

    msg = (
        f"EXIT │ {result} │ {direction} │ "
        f"Entry: {entry_price:.2f} → Exit: {exit_price:.2f} │ "
        f"PnL: {pnl_symbol}{pnl:.4f} ({pnl_symbol}{pnl_pct:.2%}) │ "
        f"Reason: {reason} │ Hold: {hold_time:.1f}s"
    )

    with LogContext(logger, trade_id=trade_id, strategy=strategy):
        if pnl >= 0:
            logger.info(msg)
        else:
            logger.warning(msg)


def log_signal(
    logger: logging.Logger,
    strategy: str,
    direction: str,
    strength: float,
    reason: str,
    **features,
) -> None:
    """
    Log a trading signal.

    Args:
        logger: Logger instance.
        strategy: Strategy name.
        direction: Signal direction.
        strength: Signal strength (0-1).
        reason: Signal reason.
        **features: Key features that triggered the signal.
    """
    feat_str = ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                         for k, v in features.items())

    msg = f"SIGNAL │ {strategy} │ {direction} │ Strength: {strength:.2f} │ {reason}"
    if feat_str:
        msg = f"{msg} │ Features: {feat_str}"

    logger.debug(msg)


def log_risk_event(
    logger: logging.Logger,
    event_type: str,
    current_value: float,
    limit_value: float,
    action: str,
) -> None:
    """
    Log a risk management event.

    Args:
        logger: Logger instance.
        event_type: Type of risk event.
        current_value: Current metric value.
        limit_value: Limit threshold.
        action: Action taken.
    """
    msg = (
        f"RISK │ {event_type} │ "
        f"Current: {current_value:.4f} │ Limit: {limit_value:.4f} │ "
        f"Action: {action}"
    )
    logger.warning(msg)
