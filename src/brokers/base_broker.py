"""Abstract BrokerClient interface.

All broker implementations (OANDA, TwelveData) must implement this interface
to ensure a consistent API for the trading system.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BrokerClient(ABC):
    """Abstract broker client interface."""

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to the broker.

        Returns:
            True if connection succeeded.
        """
        ...

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the broker."""
        ...

    @abstractmethod
    def ensure_session(self) -> None:
        """Ensure the session is still alive, reconnecting if needed."""
        ...

    @abstractmethod
    def get_account_info(self) -> dict:
        """Get account information.

        Returns:
            Dict with keys: balance, available, unrealized_pl, nav, currency.
        """
        ...

    @abstractmethod
    def get_historical_prices(
        self, instrument: str, resolution: str, count: int = 500
    ) -> pd.DataFrame:
        """Get historical OHLCV data.

        Args:
            instrument: Currency pair (e.g., 'USD_JPY').
            resolution: Candle resolution (e.g., 'HOUR_4').
            count: Number of candles to retrieve.

        Returns:
            DataFrame with columns: open, high, low, close, volume.
        """
        ...

    @abstractmethod
    def get_market_info(self, instrument: str) -> dict:
        """Get current market information.

        Args:
            instrument: Currency pair.

        Returns:
            Dict with bid, offer, spread, market_status.
        """
        ...

    @abstractmethod
    def get_positions(self) -> list[dict]:
        """Get open positions from the broker.

        Returns:
            List of dicts with: deal_id, instrument, direction, size,
            entry_price, unrealized_pl.
        """
        ...

    @abstractmethod
    def get_open_positions(self) -> pd.DataFrame:
        """Get open positions as a DataFrame.

        Returns:
            DataFrame of open positions.
        """
        ...

    @abstractmethod
    def open_position(
        self,
        epic: str,
        direction: str,
        size: float,
        order_type: str = "MARKET",
        stop_loss: float = 0,
        take_profit: float = 0,
        trailing_stop: bool = False,
    ) -> dict:
        """Open a new position.

        Args:
            epic: Instrument identifier.
            direction: 'BUY' or 'SELL'.
            size: Position size in lots.
            order_type: Order type ('MARKET').
            stop_loss: Stop-loss price.
            take_profit: Take-profit price.
            trailing_stop: Enable trailing stop.

        Returns:
            Dict with success, deal_id, level.
        """
        ...

    @abstractmethod
    def close_position(self, deal_id: str) -> float | None:
        """Close an existing position.

        Args:
            deal_id: Position identifier.

        Returns:
            Realized P&L or None on failure.
        """
        ...

    @abstractmethod
    def update_position(
        self,
        deal_id: str,
        stop_level: float | None = None,
        limit_level: float | None = None,
        epic: str = "",
    ) -> bool:
        """Update SL/TP for an existing position.

        Args:
            deal_id: Position identifier.
            stop_level: New stop-loss level.
            limit_level: New take-profit level.
            epic: Instrument (for paper trading).

        Returns:
            True if update succeeded.
        """
        ...
