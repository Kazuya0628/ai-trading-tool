"""Logging setup for the trading system."""

import sys
from pathlib import Path

from loguru import logger


def setup_logging(
    level: str = "INFO",
    log_file: str = "logs/trading.log",
    trade_log: str = "logs/trades.log",
) -> None:
    """Configure loguru logging for the trading system.

    Args:
        level: Logging level.
        log_file: Path to main log file.
        trade_log: Path to trade-specific log file.
    """
    # Remove default handler
    logger.remove()

    # Console output with color
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
               "<level>{message}</level>",
        level=level,
        colorize=True,
    )

    # Main log file
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        str(log_path),
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
        level=level,
        rotation="50 MB",
        retention="1 month",
        compression="gz",
    )

    # Trade-specific log
    trade_path = Path(trade_log)
    trade_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        str(trade_path),
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {message}",
        level="INFO",
        rotation="1 week",
        retention="3 months",
        filter=lambda record: "TRADE" in record["extra"].get("tags", []),
    )

    logger.info("Logging system initialized")


def log_trade(message: str) -> None:
    """Log a trade-specific message.

    Args:
        message: Trade log message.
    """
    logger.bind(tags=["TRADE"]).info(message)
