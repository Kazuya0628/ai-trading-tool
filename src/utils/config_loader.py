"""Configuration loader for the trading system."""

from __future__ import annotations

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
            Path(__file__).parent.parent.parent / "config" / "trading_config.yaml"
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
    env_path = Path(__file__).parent.parent.parent / ".env"
    load_dotenv(env_path)

    env_vars = {
        # Data source selection
        "DATA_SOURCE": os.getenv("DATA_SOURCE", "twelvedata"),
        # Twelve Data
        "TWELVEDATA_API_KEY": os.getenv("TWELVEDATA_API_KEY", ""),
        # OANDA (for future use)
        "OANDA_API_TOKEN": os.getenv("OANDA_API_TOKEN", ""),
        "OANDA_ACCOUNT_ID": os.getenv("OANDA_ACCOUNT_ID", ""),
        "OANDA_ENVIRONMENT": os.getenv("OANDA_ENVIRONMENT", "practice"),
        # AI
        "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY", ""),
        # Trading
        "TRADING_MODE": os.getenv("TRADING_MODE", "paper"),
        # Notifications
        "DISCORD_WEBHOOK_URL": os.getenv("DISCORD_WEBHOOK_URL", ""),
        "LINE_NOTIFY_TOKEN": os.getenv("LINE_NOTIFY_TOKEN", ""),
        # LINE Messaging API
        "LINE_CHANNEL_ACCESS_TOKEN": os.getenv("LINE_CHANNEL_ACCESS_TOKEN", ""),
        "LINE_USER_ID": os.getenv("LINE_USER_ID", ""),
        "IMGBB_API_KEY": os.getenv("IMGBB_API_KEY", ""),
    }

    # Validate critical keys based on data source
    data_source = env_vars["DATA_SOURCE"]
    if data_source == "twelvedata":
        if not env_vars["TWELVEDATA_API_KEY"]:
            logger.warning("Missing environment variable: TWELVEDATA_API_KEY")
    elif data_source == "oanda":
        missing = [k for k in ["OANDA_API_TOKEN", "OANDA_ACCOUNT_ID"] if not env_vars[k]]
        if missing:
            logger.warning(f"Missing environment variables: {missing}")

    return env_vars


def create_broker(env: dict[str, str]) -> Any:
    """Create a broker client based on DATA_SOURCE setting.

    Args:
        env: Environment variables dictionary.

    Returns:
        Broker client instance (TwelveDataClient or OandaClient).
    """
    data_source = env.get("DATA_SOURCE", "twelvedata")

    if data_source == "oanda":
        from src.brokers.oanda_client import OandaClient
        return OandaClient(
            api_token=env["OANDA_API_TOKEN"],
            account_id=env["OANDA_ACCOUNT_ID"],
            environment=env.get("OANDA_ENVIRONMENT", "practice"),
            trading_mode=env.get("TRADING_MODE", "paper"),
        )
    else:
        from src.brokers.twelvedata_client import TwelveDataClient
        return TwelveDataClient(
            api_key=env["TWELVEDATA_API_KEY"],
        )


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
    def data_source(self) -> str:
        return self.env.get("DATA_SOURCE", "twelvedata")

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
            pairs[p["instrument"]] = p
        return pairs

    @property
    def is_live(self) -> bool:
        return self.env.get("TRADING_MODE", "paper") == "live"

    @property
    def confidence_threshold(self) -> int:
        return int(self.risk_management.get("confidence_threshold", 55))

    @confidence_threshold.setter
    def confidence_threshold(self, value: int) -> None:
        rm = self.config.setdefault("risk_management", {})
        lo = int(rm.get("confidence_threshold_min", 50))
        hi = int(rm.get("confidence_threshold_max", 65))
        rm["confidence_threshold"] = max(lo, min(hi, value))

    # --------------------------------------------------
    # DESIGN-001 v1.3 new config sections
    # --------------------------------------------------

    @property
    def app(self) -> dict[str, Any]:
        """Application-level settings (mode, phase, timezone)."""
        return self.config.get("app", {})

    @property
    def phase(self) -> int:
        """Current operational phase (1/2/3)."""
        return int(self.app.get("phase", 1))

    @property
    def account_currency(self) -> str:
        return self.app.get("account_currency", "JPY")

    @property
    def broker_config(self) -> dict[str, Any]:
        """Broker provider settings."""
        return self.config.get("broker", {})

    @property
    def data_config(self) -> dict[str, Any]:
        """Data source and warmup settings."""
        return self.config.get("data", {})

    @property
    def pairs_config(self) -> dict[str, Any]:
        """New-format pairs section (active/registered lists)."""
        return self.config.get("pairs", {})

    @property
    def active_pairs_v2(self) -> list[str]:
        """Active pairs from new config format, fallback to legacy."""
        pairs_new = self.pairs_config.get("active", [])
        if pairs_new:
            return pairs_new
        return self.active_pairs  # fallback to legacy

    @property
    def scheduler_config(self) -> dict[str, Any]:
        """Scheduler layer 0/1/2 timing settings."""
        return self.config.get("scheduler", {})

    @property
    def strategy_config(self) -> dict[str, Any]:
        """Strategy settings (pattern_score_min, structural_rr_min)."""
        return self.config.get("strategy", {})

    @property
    def ai_config(self) -> dict[str, Any]:
        """AI configuration (gemini + groq sections)."""
        return self.config.get("ai", {})

    @property
    def gemini_config(self) -> dict[str, Any]:
        """Gemini AI settings."""
        return self.ai_config.get("gemini", {})

    @property
    def groq_config(self) -> dict[str, Any]:
        """Groq AI settings."""
        return self.ai_config.get("groq", {})

    @property
    def consensus_config(self) -> dict[str, Any]:
        """Consensus engine thresholds."""
        return self.config.get("consensus", {})

    @property
    def execution_config(self) -> dict[str, Any]:
        """Execution settings (slippage, expiry, spread)."""
        return self.config.get("execution", {})

    @property
    def portfolio_config(self) -> dict[str, Any]:
        """Portfolio settings (NAV anchors)."""
        return self.config.get("portfolio", {})

    @property
    def dashboard_config(self) -> dict[str, Any]:
        """Dashboard settings (host, port, refresh)."""
        return self.config.get("dashboard", {})

    @property
    def notifications_config(self) -> dict[str, Any]:
        """Notification settings."""
        return self.config.get("notifications", {})
