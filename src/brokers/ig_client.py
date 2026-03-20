"""IG Securities API Client - Full broker integration.

Provides a high-level wrapper around the trading-ig library
for IG Securities REST API operations.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
from loguru import logger
from trading_ig import IGService
from trading_ig.rest import IGException


class IGClient:
    """High-level client for IG Securities REST API.

    Handles session management, market data retrieval,
    order execution, and position management.
    """

    def __init__(
        self,
        api_key: str,
        username: str,
        password: str,
        acc_type: str = "DEMO",
        acc_number: str = "",
    ) -> None:
        """Initialize IG Securities client.

        Args:
            api_key: IG API key.
            username: IG account username.
            password: IG account password.
            acc_type: 'DEMO' or 'LIVE'.
            acc_number: Account number for sub-account switching.
        """
        self.api_key = api_key
        self.username = username
        self.password = password
        self.acc_type = acc_type
        self.acc_number = acc_number
        self._service: IGService | None = None
        self._session_time: datetime | None = None
        self._session_timeout = timedelta(minutes=55)

    # --------------------------------------------------
    # Session Management
    # --------------------------------------------------

    def connect(self) -> bool:
        """Establish connection to IG Securities API.

        Returns:
            True if connection successful.
        """
        try:
            self._service = IGService(
                self.username,
                self.password,
                self.api_key,
                self.acc_type,
            )
            self._service.create_session()
            self._session_time = datetime.now()

            if self.acc_number:
                self._service.switch_account(self.acc_number, False)

            logger.info(
                f"Connected to IG Securities ({self.acc_type}) "
                f"as {self.username}"
            )
            return True

        except IGException as e:
            logger.error(f"IG connection failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected connection error: {e}")
            return False

    def disconnect(self) -> None:
        """Close the IG session."""
        if self._service:
            try:
                self._service.logout()
                logger.info("Disconnected from IG Securities")
            except Exception as e:
                logger.warning(f"Disconnect warning: {e}")
            finally:
                self._service = None
                self._session_time = None

    def ensure_session(self) -> None:
        """Ensure the session is active, refresh if needed."""
        if self._service is None:
            self.connect()
            return

        if self._session_time and datetime.now() - self._session_time > self._session_timeout:
            logger.info("Session timeout approaching, refreshing...")
            self.disconnect()
            self.connect()

    @property
    def service(self) -> IGService:
        """Get the active IGService instance."""
        self.ensure_session()
        if self._service is None:
            raise ConnectionError("Not connected to IG Securities")
        return self._service

    # --------------------------------------------------
    # Account Information
    # --------------------------------------------------

    def get_account_info(self) -> dict[str, Any]:
        """Get account balance and details.

        Returns:
            Account information dictionary.
        """
        try:
            accounts = self.service.fetch_accounts()
            if accounts is not None and not accounts.empty:
                account = accounts.iloc[0]
                return {
                    "account_id": account.get("accountId", ""),
                    "account_name": account.get("accountName", ""),
                    "balance": float(account.get("balance", 0)),
                    "deposit": float(account.get("deposit", 0)),
                    "profit_loss": float(account.get("profitLoss", 0)),
                    "available": float(account.get("available", 0)),
                    "currency": account.get("currency", "JPY"),
                }
            return {}
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            return {}

    # --------------------------------------------------
    # Market Data
    # --------------------------------------------------

    def search_markets(self, term: str) -> pd.DataFrame:
        """Search for markets by keyword.

        Args:
            term: Search term (e.g., 'USD/JPY', 'EUR/USD').

        Returns:
            DataFrame of matching markets.
        """
        try:
            result = self.service.search_markets(term)
            return result if result is not None else pd.DataFrame()
        except Exception as e:
            logger.error(f"Market search failed for '{term}': {e}")
            return pd.DataFrame()

    def get_market_info(self, epic: str) -> dict[str, Any]:
        """Get detailed market information for an epic.

        Args:
            epic: IG market identifier (e.g., 'CS.D.USDJPY.TODAY.IP').

        Returns:
            Market details dictionary.
        """
        try:
            market = self.service.fetch_market_by_epic(epic)
            if market is not None:
                instrument = market.get("instrument", {})
                snapshot = market.get("snapshot", {})
                dealing = market.get("dealingRules", {})
                return {
                    "epic": epic,
                    "name": instrument.get("name", ""),
                    "type": instrument.get("type", ""),
                    "bid": float(snapshot.get("bid", 0)),
                    "offer": float(snapshot.get("offer", 0)),
                    "high": float(snapshot.get("high", 0)),
                    "low": float(snapshot.get("low", 0)),
                    "spread": float(snapshot.get("offer", 0)) - float(snapshot.get("bid", 0)),
                    "market_status": snapshot.get("marketStatus", ""),
                    "min_deal_size": dealing.get("minDealSize", {}).get("value", 0),
                    "lot_size": instrument.get("lotSize", 1),
                    "currency": instrument.get("currencies", [{}])[0].get("code", "JPY")
                    if instrument.get("currencies") else "JPY",
                }
            return {}
        except Exception as e:
            logger.error(f"Failed to get market info for {epic}: {e}")
            return {}

    def get_historical_prices(
        self,
        epic: str,
        resolution: str = "HOUR_4",
        num_points: int = 500,
    ) -> pd.DataFrame:
        """Fetch historical OHLCV price data.

        Args:
            epic: Market epic identifier.
            resolution: Candle resolution (e.g., 'HOUR', 'HOUR_4', 'DAY').
            num_points: Number of data points to fetch.

        Returns:
            DataFrame with OHLCV data.
        """
        try:
            response = self.service.fetch_historical_prices_by_epic_and_num_points(
                epic, resolution, num_points
            )

            if response and "prices" in response:
                prices = response["prices"]
                # Build unified OHLCV DataFrame
                df = pd.DataFrame({
                    "open": (prices["bid"]["Open"] + prices["ask"]["Open"]) / 2,
                    "high": (prices["bid"]["High"] + prices["ask"]["High"]) / 2,
                    "low": (prices["bid"]["Low"] + prices["ask"]["Low"]) / 2,
                    "close": (prices["bid"]["Close"] + prices["ask"]["Close"]) / 2,
                    "volume": prices.get("last", {}).get("Volume", pd.Series(0, index=prices["bid"].index)),
                })
                df.index = pd.to_datetime(df.index)
                df.index.name = "datetime"
                return df

            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Failed to fetch prices for {epic}: {e}")
            return pd.DataFrame()

    def get_historical_prices_by_date(
        self,
        epic: str,
        resolution: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Fetch historical prices within a date range.

        Args:
            epic: Market epic identifier.
            resolution: Candle resolution.
            start_date: Start date string (YYYY-MM-DD).
            end_date: End date string (YYYY-MM-DD).

        Returns:
            DataFrame with OHLCV data.
        """
        try:
            response = self.service.fetch_historical_prices_by_epic_and_date_range(
                epic, resolution, start_date, end_date
            )

            if response and "prices" in response:
                prices = response["prices"]
                df = pd.DataFrame({
                    "open": (prices["bid"]["Open"] + prices["ask"]["Open"]) / 2,
                    "high": (prices["bid"]["High"] + prices["ask"]["High"]) / 2,
                    "low": (prices["bid"]["Low"] + prices["ask"]["Low"]) / 2,
                    "close": (prices["bid"]["Close"] + prices["ask"]["Close"]) / 2,
                    "volume": prices.get("last", {}).get("Volume", pd.Series(0, index=prices["bid"].index)),
                })
                df.index = pd.to_datetime(df.index)
                df.index.name = "datetime"
                return df

            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Failed to fetch date-range prices for {epic}: {e}")
            return pd.DataFrame()

    # --------------------------------------------------
    # Position Management
    # --------------------------------------------------

    def get_open_positions(self) -> pd.DataFrame:
        """Fetch all open positions.

        Returns:
            DataFrame of open positions.
        """
        try:
            positions = self.service.fetch_open_positions()
            return positions if positions is not None else pd.DataFrame()
        except Exception as e:
            logger.error(f"Failed to fetch open positions: {e}")
            return pd.DataFrame()

    def get_working_orders(self) -> pd.DataFrame:
        """Fetch all working (pending) orders.

        Returns:
            DataFrame of working orders.
        """
        try:
            orders = self.service.fetch_working_orders()
            return orders if orders is not None else pd.DataFrame()
        except Exception as e:
            logger.error(f"Failed to fetch working orders: {e}")
            return pd.DataFrame()

    # --------------------------------------------------
    # Order Execution
    # --------------------------------------------------

    def open_position(
        self,
        epic: str,
        direction: str,
        size: float,
        order_type: str = "MARKET",
        currency_code: str = "JPY",
        stop_loss: float | None = None,
        stop_distance: float | None = None,
        take_profit: float | None = None,
        limit_distance: float | None = None,
        guaranteed_stop: bool = False,
        trailing_stop: bool = False,
        trailing_stop_increment: float | None = None,
    ) -> dict[str, Any]:
        """Open a new position.

        Args:
            epic: Market epic identifier.
            direction: 'BUY' or 'SELL'.
            size: Position size in lots.
            order_type: 'MARKET' or 'LIMIT'.
            currency_code: Account currency.
            stop_loss: Absolute stop-loss price level.
            stop_distance: Stop distance in points.
            take_profit: Absolute take-profit price level.
            limit_distance: Limit (TP) distance in points.
            guaranteed_stop: Use guaranteed stop-loss.
            trailing_stop: Enable trailing stop.
            trailing_stop_increment: Trailing stop increment in points.

        Returns:
            Deal confirmation dictionary.
        """
        try:
            logger.info(
                f"Opening {direction} position: {epic}, size={size}, "
                f"SL={stop_loss or stop_distance}, TP={take_profit or limit_distance}"
            )

            result = self.service.create_open_position(
                epic=epic,
                direction=direction,
                currency_code=currency_code,
                order_type=order_type,
                size=size,
                level=None,
                stop_level=stop_loss,
                stop_distance=stop_distance,
                limit_level=take_profit,
                limit_distance=limit_distance,
                guaranteed_stop=guaranteed_stop,
                trailing_stop=trailing_stop,
                trailing_stop_increment=trailing_stop_increment,
                expiry="-",
                force_open=True,
            )

            deal_ref = result.get("dealReference", "")
            if deal_ref:
                # Confirm the deal
                confirm = self.service.fetch_deal_by_deal_reference(deal_ref)
                deal_id = confirm.get("dealId", "")
                deal_status = confirm.get("dealStatus", "")
                reason = confirm.get("reason", "")

                if deal_status == "ACCEPTED":
                    logger.info(f"Position opened: deal_id={deal_id}")
                    return {
                        "success": True,
                        "deal_id": deal_id,
                        "deal_ref": deal_ref,
                        "status": deal_status,
                        "direction": direction,
                        "epic": epic,
                        "size": size,
                        "level": confirm.get("level", 0),
                        "stop_level": confirm.get("stopLevel", 0),
                        "limit_level": confirm.get("limitLevel", 0),
                    }
                else:
                    logger.warning(f"Deal rejected: {reason}")
                    return {
                        "success": False,
                        "deal_ref": deal_ref,
                        "status": deal_status,
                        "reason": reason,
                    }
            return {"success": False, "reason": "No deal reference returned"}

        except Exception as e:
            logger.error(f"Failed to open position: {e}")
            return {"success": False, "reason": str(e)}

    def close_position(
        self,
        deal_id: str,
        direction: str,
        size: float,
        order_type: str = "MARKET",
    ) -> dict[str, Any]:
        """Close an existing position.

        Args:
            deal_id: The deal ID of the position to close.
            direction: Opposite direction ('BUY' to close SELL, 'SELL' to close BUY).
            size: Position size to close.
            order_type: 'MARKET' or 'LIMIT'.

        Returns:
            Close confirmation dictionary.
        """
        try:
            logger.info(f"Closing position: deal_id={deal_id}, direction={direction}")

            result = self.service.close_open_position(
                deal_id=deal_id,
                direction=direction,
                size=size,
                order_type=order_type,
                level=None,
                expiry="-",
            )

            deal_ref = result.get("dealReference", "")
            if deal_ref:
                confirm = self.service.fetch_deal_by_deal_reference(deal_ref)
                deal_status = confirm.get("dealStatus", "")
                reason = confirm.get("reason", "")

                if deal_status == "ACCEPTED":
                    logger.info(f"Position closed: deal_id={deal_id}")
                    return {
                        "success": True,
                        "deal_id": deal_id,
                        "status": deal_status,
                    }
                else:
                    logger.warning(f"Close rejected: {reason}")
                    return {"success": False, "reason": reason}

            return {"success": False, "reason": "No deal reference"}

        except Exception as e:
            logger.error(f"Failed to close position {deal_id}: {e}")
            return {"success": False, "reason": str(e)}

    def update_position(
        self,
        deal_id: str,
        stop_level: float | None = None,
        limit_level: float | None = None,
        trailing_stop: bool = False,
        trailing_stop_distance: float | None = None,
        trailing_stop_increment: float | None = None,
    ) -> dict[str, Any]:
        """Update stop-loss and take-profit for an open position.

        Args:
            deal_id: Deal ID to update.
            stop_level: New stop-loss price level.
            limit_level: New take-profit price level.
            trailing_stop: Enable trailing stop.
            trailing_stop_distance: Trailing stop distance.
            trailing_stop_increment: Trailing stop increment.

        Returns:
            Update confirmation dictionary.
        """
        try:
            result = self.service.update_open_position(
                deal_id=deal_id,
                stop_level=stop_level,
                limit_level=limit_level,
                trailing_stop=trailing_stop,
                trailing_stop_distance=trailing_stop_distance,
                trailing_stop_increment=trailing_stop_increment,
            )

            deal_ref = result.get("dealReference", "")
            if deal_ref:
                confirm = self.service.fetch_deal_by_deal_reference(deal_ref)
                if confirm.get("dealStatus") == "ACCEPTED":
                    logger.info(f"Position updated: {deal_id}")
                    return {"success": True, "deal_id": deal_id}

            return {"success": False, "reason": "Update failed"}

        except Exception as e:
            logger.error(f"Failed to update position {deal_id}: {e}")
            return {"success": False, "reason": str(e)}

    # --------------------------------------------------
    # Working Orders
    # --------------------------------------------------

    def place_limit_order(
        self,
        epic: str,
        direction: str,
        size: float,
        level: float,
        currency_code: str = "JPY",
        stop_distance: float | None = None,
        limit_distance: float | None = None,
        good_till_date: str | None = None,
    ) -> dict[str, Any]:
        """Place a limit order (pending order).

        Args:
            epic: Market epic.
            direction: 'BUY' or 'SELL'.
            size: Position size.
            level: Entry price level.
            currency_code: Account currency.
            stop_distance: Stop distance in points.
            limit_distance: Take-profit distance in points.
            good_till_date: Expiry date (ISO format) or None for GTC.

        Returns:
            Order confirmation dictionary.
        """
        try:
            result = self.service.create_working_order(
                epic=epic,
                direction=direction,
                size=size,
                level=level,
                currency_code=currency_code,
                order_type="LIMIT",
                stop_distance=stop_distance,
                limit_distance=limit_distance,
                good_till_date=good_till_date,
                guaranteed_stop=False,
                expiry="-",
                force_open=True,
            )

            deal_ref = result.get("dealReference", "")
            if deal_ref:
                confirm = self.service.fetch_deal_by_deal_reference(deal_ref)
                if confirm.get("dealStatus") == "ACCEPTED":
                    return {
                        "success": True,
                        "deal_id": confirm.get("dealId", ""),
                    }
            return {"success": False, "reason": "Order placement failed"}

        except Exception as e:
            logger.error(f"Failed to place limit order: {e}")
            return {"success": False, "reason": str(e)}

    def cancel_order(self, deal_id: str) -> dict[str, Any]:
        """Cancel a working order.

        Args:
            deal_id: The deal ID of the order to cancel.

        Returns:
            Cancellation confirmation.
        """
        try:
            result = self.service.delete_working_order(deal_id)
            deal_ref = result.get("dealReference", "")
            if deal_ref:
                confirm = self.service.fetch_deal_by_deal_reference(deal_ref)
                if confirm.get("dealStatus") == "ACCEPTED":
                    return {"success": True, "deal_id": deal_id}
            return {"success": False, "reason": "Cancel failed"}
        except Exception as e:
            logger.error(f"Failed to cancel order {deal_id}: {e}")
            return {"success": False, "reason": str(e)}

    # --------------------------------------------------
    # Market Sentiment
    # --------------------------------------------------

    def get_client_sentiment(self, market_id: str) -> dict[str, Any]:
        """Get client sentiment data for a market.

        Args:
            market_id: Market ID for sentiment lookup.

        Returns:
            Sentiment data dictionary.
        """
        try:
            sentiment = self.service.fetch_client_sentiment_by_instrument(market_id)
            if sentiment:
                return {
                    "long_pct": float(sentiment.get("longPositionPercentage", 50)),
                    "short_pct": float(sentiment.get("shortPositionPercentage", 50)),
                    "market_id": market_id,
                }
            return {"long_pct": 50, "short_pct": 50}
        except Exception as e:
            logger.error(f"Failed to get sentiment: {e}")
            return {"long_pct": 50, "short_pct": 50}
