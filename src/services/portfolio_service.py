"""Portfolio Service — NAV, drawdown, and exposure management.

Responsibilities:
- NAV / DD / Exposure calculation
- Peak NAV management
- Daily/Weekly anchor management
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from loguru import logger


@dataclass
class PortfolioState:
    """Current portfolio state snapshot."""

    nav: float = 0.0
    balance: float = 0.0
    unrealized_pl: float = 0.0
    peak_nav: float = 0.0
    current_drawdown_pct: float = 0.0
    daily_realized_pl: float = 0.0
    weekly_realized_pl: float = 0.0
    open_position_count: int = 0
    exposure_by_currency: dict[str, int] = field(default_factory=dict)
    last_daily_reset: str = ""
    last_weekly_reset: str = ""


class PortfolioService:
    """Manages portfolio-level state and risk metrics."""

    MAX_CURRENCY_EXPOSURE: int = 3  # Max positions with same base/quote currency

    def __init__(self) -> None:
        self.state = PortfolioState()

    def update_from_broker(self, account_info: dict[str, Any]) -> None:
        """Update portfolio state from broker account info.

        Args:
            account_info: Dict with balance, unrealized_pl, nav.
        """
        self.state.balance = account_info.get("balance", 0)
        self.state.unrealized_pl = account_info.get("unrealized_pl", 0)
        self.state.nav = account_info.get("nav", self.state.balance + self.state.unrealized_pl)

        # Update peak NAV
        if self.state.nav > self.state.peak_nav:
            self.state.peak_nav = self.state.nav

        # Update drawdown
        if self.state.peak_nav > 0:
            self.state.current_drawdown_pct = (
                (self.state.peak_nav - self.state.nav) / self.state.peak_nav * 100
            )

    def update_positions(self, positions: list[dict[str, Any]]) -> None:
        """Update position count and currency exposure.

        Args:
            positions: List of open position dicts.
        """
        self.state.open_position_count = len(positions)

        # Calculate currency exposure
        exposure: dict[str, int] = {}
        for pos in positions:
            instrument = pos.get("instrument", "")
            if "_" in instrument:
                base, quote = instrument.split("_", 1)
                exposure[base] = exposure.get(base, 0) + 1
                exposure[quote] = exposure.get(quote, 0) + 1
        self.state.exposure_by_currency = exposure

    def record_trade_pnl(self, pnl: float) -> None:
        """Record a closed trade's P&L.

        Args:
            pnl: Realized profit/loss.
        """
        self._check_resets()
        self.state.daily_realized_pl += pnl
        self.state.weekly_realized_pl += pnl

    def check_currency_exposure(self, instrument: str) -> tuple[bool, str]:
        """Check if adding a position violates currency exposure limits.

        Args:
            instrument: Currency pair to check.

        Returns:
            Tuple of (allowed, reason).
        """
        if "_" not in instrument:
            return True, "OK"

        base, quote = instrument.split("_", 1)
        for currency in (base, quote):
            current = self.state.exposure_by_currency.get(currency, 0)
            if current >= self.MAX_CURRENCY_EXPOSURE:
                return False, (
                    f"Currency {currency} exposure at limit: "
                    f"{current}/{self.MAX_CURRENCY_EXPOSURE}"
                )
        return True, "OK"

    def get_snapshot(self) -> dict[str, Any]:
        """Get current portfolio snapshot for dashboard.

        Returns:
            Portfolio state as dict.
        """
        return {
            "nav": self.state.nav,
            "balance": self.state.balance,
            "unrealized_pl": self.state.unrealized_pl,
            "peak_nav": self.state.peak_nav,
            "drawdown_pct": round(self.state.current_drawdown_pct, 2),
            "daily_realized_pl": self.state.daily_realized_pl,
            "weekly_realized_pl": self.state.weekly_realized_pl,
            "open_positions": self.state.open_position_count,
            "currency_exposure": dict(self.state.exposure_by_currency),
        }

    def _check_resets(self) -> None:
        """Reset daily/weekly counters on date/week rollover."""
        today = datetime.now().strftime("%Y-%m-%d")
        if self.state.last_daily_reset != today:
            self.state.daily_realized_pl = 0.0
            self.state.last_daily_reset = today

        week_key = datetime.now().strftime("%Y-W%W")
        if self.state.last_weekly_reset != week_key:
            self.state.weekly_realized_pl = 0.0
            self.state.last_weekly_reset = week_key
