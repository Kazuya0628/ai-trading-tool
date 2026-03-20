"""Configuration loader for the trading system."""

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from loguru import logger


def load_config(config_path: str | None = None) -> dict[str, Any]:
    """Load trading configuration from YAML file.

    Args:
        config_path: Path to config YAML. Defaults to config/trading_config.yaml.

    Returns:
        Configuration dictionary.
    """
    if config_path is None:
        config_path = str(
            Path(__file__).parent.parent / "config" / "trading_config.yaml"
        )

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info(f"Configuration loaded from {config_path}")
    return config


def load_env() -> dict[str, str]:
    """Load environment variables from .env file.

    Returns:
        Dictionary of environment variables.
    """
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(env_path)

    env_vars = {
        "IG_API_KEY": os.getenv("IG_API_KEY", ""),
        "IG_USERNAME": os.getenv("IG_USERNAME", ""),
        "IG_PASSWORD": os.getenv("IG_PASSWORD", ""),
        "IG_ACC_TYPE": os.getenv("IG_ACC_TYPE", "DEMO"),
        "IG_ACC_NUMBER": os.getenv("IG_ACC_NUMBER", ""),
        "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY", ""),
        "TRADING_MODE": os.getenv("TRADING_MODE", "paper"),
        "DISCORD_WEBHOOK_URL": os.getenv("DISCORD_WEBHOOK_URL", ""),
        "LINE_NOTIFY_TOKEN": os.getenv("LINE_NOTIFY_TOKEN", ""),
    }

    # Validate critical keys
    missing = [k for k in ["IG_API_KEY", "IG_USERNAME", "IG_PASSWORD"] if not env_vars[k]]
    if missing:
        logger.warning(f"Missing environment variables: {missing}")

    return env_vars


class TradingConfig:
    """Centralized configuration manager for the trading system."""

    def __init__(self, config_path: str | None = None) -> None:
        self.env = load_env()
        self.config = load_config(config_path)
        self._validate()

    def _validate(self) -> None:
        """Validate configuration consistency."""
        risk = self.risk_management
        if risk["risk_per_trade_pct"] > 5.0:
            logger.warning("Risk per trade > 5% is extremely dangerous!")
        if risk["max_drawdown_pct"] > 20.0:
            logger.warning("Max drawdown > 20% is very high!")

    @property
    def trading(self) -> dict[str, Any]:
        return self.config.get("trading", {})

    @property
    def timeframes(self) -> dict[str, str]:
        return self.config.get("timeframes", {})

    @property
    def pattern_recognition(self) -> dict[str, Any]:
        return self.config.get("pattern_recognition", {})

    @property
    def indicators(self) -> dict[str, Any]:
        return self.config.get("indicators", {})

    @property
    def ai_analysis(self) -> dict[str, Any]:
        return self.config.get("ai_analysis", {})

    @property
    def risk_management(self) -> dict[str, Any]:
        return self.config.get("risk_management", {})

    @property
    def backtesting(self) -> dict[str, Any]:
        return self.config.get("backtesting", {})

    @property
    def execution(self) -> dict[str, Any]:
        return self.config.get("execution", {})

    @property
    def active_pairs(self) -> list[str]:
        return self.trading.get("active_pairs", [])

    @property
    def pair_configs(self) -> dict[str, dict]:
        pairs = {}
        for p in self.trading.get("pairs", []):
            pairs[p["epic"]] = p
        return pairs

    @property
    def is_live(self) -> bool:
        return self.env.get("TRADING_MODE", "paper") == "live"

    @property
    def is_demo(self) -> bool:
        return self.env.get("IG_ACC_TYPE", "DEMO") == "DEMO"
