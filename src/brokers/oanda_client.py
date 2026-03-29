"""OANDA v20 API Client - Market data and paper trading.

Provides a high-level wrapper around the OANDA v20 REST API
for fetching forex market data and simulating paper trades.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import requests
from loguru import logger


# OANDA granularity mapping from IG-style resolution names
_GRANULARITY_MAP = {
    "SECOND": "S5",
    "MINUTE": "M1",
    "MINUTE_2": "M2",
    "MINUTE_3": "M4",      # closest available
    "MINUTE_5": "M5",
    "MINUTE_10": "M10",
    "MINUTE_15": "M15",
    "MINUTE_30": "M30",
    "HOUR": "H1",
    "HOUR_2": "H2",
    "HOUR_3": "H3",
    "HOUR_4": "H4",
    "HOUR_6": "H6",
    "HOUR_8": "H8",
    "HOUR_12": "H12",
    "DAY": "D",
    "WEEK": "W",
    "MONTH": "M",
    # Also accept OANDA native granularities directly
    "S5": "S5", "M1": "M1", "M2": "M2", "M4": "M4", "M5": "M5",
    "M10": "M10", "M15": "M15", "M30": "M30",
    "H1": "H1", "H2": "H2", "H3": "H3", "H4": "H4",
    "H6": "H6", "H8": "H8", "H12": "H12",
    "D": "D", "W": "W", "M": "M",
}


class OandaClient:
    """High-level client for OANDA v20 REST API.

    Handles market data retrieval and order execution.
    Supports both paper trading (TRADING_MODE=paper) and
    live trading via OANDA v20 API (TRADING_MODE=live).
    """

    PRACTICE_URL = "https://api-fxpractice.oanda.com"
    LIVE_URL = "https://api-fxtrade.oanda.com"

    def __init__(
        self,
        api_token: str,
        account_id: str,
        environment: str = "practice",
        trading_mode: str = "paper",
    ) -> None:
        """Initialize OANDA client.

        Args:
            api_token: OANDA v20 API access token.
            account_id: OANDA account ID.
            environment: 'practice' or 'live'.
            trading_mode: 'paper' (simulate) or 'live' (real OANDA orders).
        """
        self.api_token = api_token
        self.account_id = account_id
        self.environment = environment
        self.trading_mode = trading_mode
        self._base_url = (
            self.PRACTICE_URL if environment == "practice" else self.LIVE_URL
        )
        self._session: requests.Session | None = None
        self._connected = False

        # Paper trading state
        self._paper_positions: dict[str, dict[str, Any]] = {}
        self._paper_balance: float = 1_000_000.0  # Default 100万円
        self._paper_pnl: float = 0.0
        self._next_deal_id: int = 1

    # --------------------------------------------------
    # Session Management
    # --------------------------------------------------

    def connect(self) -> bool:
        """Establish connection to OANDA API.

        Returns:
            True if connection successful.
        """
        try:
            self._session = requests.Session()
            self._session.headers.update({
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json",
                "Accept-Datetime-Format": "RFC3339",
            })

            # Validate connection by fetching account summary
            resp = self._request("GET", f"/v3/accounts/{self.account_id}/summary")
            if resp and "account" in resp:
                account = resp["account"]
                self._paper_balance = float(account.get("balance", self._paper_balance))
                self._connected = True
                logger.info(
                    f"Connected to OANDA ({self.environment}) "
                    f"account={self.account_id}"
                )
                return True

            logger.error("OANDA connection failed: invalid response")
            return False

        except Exception as e:
            logger.error(f"OANDA connection error: {e}")
            return False

    def disconnect(self) -> None:
        """Close the OANDA session."""
        if self._session:
            self._session.close()
            self._session = None
        self._connected = False
        logger.info("Disconnected from OANDA")

    def ensure_session(self) -> None:
        """Ensure the session is active."""
        if not self._connected or self._session is None:
            self.connect()

    # --------------------------------------------------
    # Account Information
    # --------------------------------------------------

    def get_account_info(self) -> dict[str, Any]:
        """Get account balance and details.

        Returns:
            Account information dictionary.
        """
        try:
            resp = self._request("GET", f"/v3/accounts/{self.account_id}/summary")
            if resp and "account" in resp:
                account = resp["account"]
                balance = float(account.get("balance", 0))
                unrealized_pl = float(account.get("unrealizedPL", 0))
                return {
                    "account_id": account.get("id", self.account_id),
                    "account_name": f"OANDA {self.environment.title()}",
                    "balance": balance,
                    "deposit": balance,
                    "profit_loss": unrealized_pl,
                    "unrealized_pl": unrealized_pl,
                    "realized_pl": float(account.get("pl", 0)),
                    "nav": float(account.get("NAV", balance)),
                    "available": float(account.get("marginAvailable", balance)),
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
        """Search for instruments by keyword.

        Args:
            term: Search term (e.g., 'USD_JPY', 'EUR').

        Returns:
            DataFrame of matching instruments.
        """
        try:
            resp = self._request(
                "GET",
                f"/v3/accounts/{self.account_id}/instruments",
            )
            if resp and "instruments" in resp:
                instruments = resp["instruments"]
                term_upper = term.upper().replace("/", "_")
                matches = [
                    {
                        "instrument": i["name"],
                        "displayName": i.get("displayName", i["name"]),
                        "type": i.get("type", ""),
                        "pipLocation": i.get("pipLocation", 0),
                    }
                    for i in instruments
                    if term_upper in i["name"]
                ]
                return pd.DataFrame(matches)
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Market search failed for '{term}': {e}")
            return pd.DataFrame()

    def get_market_info(self, instrument: str) -> dict[str, Any]:
        """Get current pricing for an instrument.

        Args:
            instrument: OANDA instrument (e.g., 'USD_JPY').

        Returns:
            Market details dictionary.
        """
        try:
            resp = self._request(
                "GET",
                f"/v3/accounts/{self.account_id}/pricing",
                params={"instruments": instrument},
            )
            if resp and "prices" in resp and resp["prices"]:
                price = resp["prices"][0]
                bids = price.get("bids", [{}])
                asks = price.get("asks", [{}])
                bid = float(bids[0].get("price", 0)) if bids else 0
                ask = float(asks[0].get("price", 0)) if asks else 0
                return {
                    "epic": instrument,
                    "name": price.get("instrument", instrument),
                    "type": "CURRENCIES",
                    "bid": bid,
                    "offer": ask,
                    "high": 0,  # Not available from pricing endpoint
                    "low": 0,
                    "spread": ask - bid,
                    "market_status": price.get("tradeable", "UNKNOWN"),
                    "min_deal_size": 1,
                    "lot_size": 1,
                    "currency": "JPY" if "JPY" in instrument else "USD",
                }
            return {}
        except Exception as e:
            logger.error(f"Failed to get market info for {instrument}: {e}")
            return {}

    def get_historical_prices(
        self,
        instrument: str,
        resolution: str = "HOUR_4",
        num_points: int = 500,
    ) -> pd.DataFrame:
        """Fetch historical OHLCV price data.

        Args:
            instrument: OANDA instrument (e.g., 'USD_JPY').
            resolution: Candle resolution (e.g., 'HOUR_4', 'DAY', 'H4').
            num_points: Number of candles to fetch.

        Returns:
            DataFrame with OHLCV data.
        """
        granularity = _GRANULARITY_MAP.get(resolution, resolution)
        if granularity not in _GRANULARITY_MAP.values():
            logger.error(f"Invalid resolution: {resolution}")
            return pd.DataFrame()

        try:
            # OANDA limits to 5000 candles per request
            count = min(num_points, 5000)
            resp = self._request(
                "GET",
                f"/v3/instruments/{instrument}/candles",
                params={
                    "granularity": granularity,
                    "count": count,
                    "price": "M",  # Mid prices
                },
            )

            if resp and "candles" in resp:
                return self._candles_to_dataframe(resp["candles"])

            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Failed to fetch prices for {instrument}: {e}")
            return pd.DataFrame()

    def get_historical_prices_by_date(
        self,
        instrument: str,
        resolution: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Fetch historical prices within a date range.

        Args:
            instrument: OANDA instrument.
            resolution: Candle resolution.
            start_date: Start date string (YYYY-MM-DD).
            end_date: End date string (YYYY-MM-DD).

        Returns:
            DataFrame with OHLCV data.
        """
        granularity = _GRANULARITY_MAP.get(resolution, resolution)
        if granularity not in _GRANULARITY_MAP.values():
            logger.error(f"Invalid resolution: {resolution}")
            return pd.DataFrame()

        try:
            from_time = f"{start_date}T00:00:00Z"
            to_time = f"{end_date}T23:59:59Z"

            all_candles: list[dict] = []
            current_from = from_time

            # Paginate if needed (max 5000 per request)
            while True:
                resp = self._request(
                    "GET",
                    f"/v3/instruments/{instrument}/candles",
                    params={
                        "granularity": granularity,
                        "from": current_from,
                        "to": to_time,
                        "price": "M",
                        "count": 5000,
                    },
                )

                if not resp or "candles" not in resp or not resp["candles"]:
                    break

                candles = resp["candles"]
                all_candles.extend(candles)

                # If we got less than 5000, we've fetched everything
                if len(candles) < 5000:
                    break

                # Move the start time to after the last candle
                last_time = candles[-1]["time"]
                current_from = last_time

            if all_candles:
                return self._candles_to_dataframe(all_candles)

            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Failed to fetch date-range prices for {instrument}: {e}")
            return pd.DataFrame()

    # --------------------------------------------------
    # Position Management (Paper Trading)
    # --------------------------------------------------

    def get_open_positions(self) -> pd.DataFrame:
        """Fetch all open positions (OANDA API for live, local for paper).

        Returns:
            DataFrame of open positions.
        """
        if self.trading_mode == "live":
            return self._get_live_positions()
        return self._get_paper_positions()

    def _get_live_positions(self) -> pd.DataFrame:
        """Fetch open trades from OANDA v20 API."""
        try:
            resp = self._request("GET", f"/v3/accounts/{self.account_id}/openTrades")
            if not resp or "trades" not in resp:
                return pd.DataFrame()

            rows = []
            for trade in resp["trades"]:
                units = float(trade.get("currentUnits", 0))
                direction = "BUY" if units > 0 else "SELL"
                size_units = abs(units)
                instrument = trade.get("instrument", "")
                # Convert OANDA units back to lots
                if "JPY" in instrument:
                    size_lots = size_units / 100
                else:
                    size_lots = size_units / 10_000

                sl_order = trade.get("stopLossOrder", {})
                tp_order = trade.get("takeProfitOrder", {})
                rows.append({
                    "dealId": f"OANDA-{trade['id']}",
                    "epic": instrument,
                    "direction": direction,
                    "size": size_lots,
                    "level": float(trade.get("price", 0)),
                    "stopLevel": float(sl_order.get("price", 0)) if sl_order else None,
                    "limitLevel": float(tp_order.get("price", 0)) if tp_order else None,
                    "openedAt": trade.get("openTime", ""),
                    "unrealizedPL": float(trade.get("unrealizedPL", 0)),
                })
            return pd.DataFrame(rows) if rows else pd.DataFrame()
        except Exception as e:
            logger.error(f"Failed to fetch live positions: {e}")
            return pd.DataFrame()

    def _get_paper_positions(self) -> pd.DataFrame:
        """Return local paper positions as DataFrame."""
        if not self._paper_positions:
            return pd.DataFrame()

        rows = []
        for instrument, pos in self._paper_positions.items():
            rows.append({
                "dealId": pos["deal_id"],
                "epic": instrument,
                "direction": pos["direction"],
                "size": pos["size"],
                "level": pos["entry_price"],
                "stopLevel": pos.get("stop_loss"),
                "limitLevel": pos.get("take_profit"),
                "openedAt": pos.get("opened_at", ""),
            })
        return pd.DataFrame(rows)

    def get_working_orders(self) -> pd.DataFrame:
        """Fetch working orders (not supported in paper mode).

        Returns:
            Empty DataFrame.
        """
        return pd.DataFrame()

    # --------------------------------------------------
    # Order Execution (Paper Trading)
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
        """Open a position (paper or live depending on trading_mode).

        Args:
            epic: Instrument identifier.
            direction: 'BUY' or 'SELL'.
            size: Position size in lots.
            order_type: Order type.
            currency_code: Account currency.
            stop_loss: Absolute stop-loss price level.
            stop_distance: Stop distance in points.
            take_profit: Absolute take-profit price level.
            limit_distance: Limit distance in points.
            guaranteed_stop: Ignored.
            trailing_stop: Enable trailing stop.
            trailing_stop_increment: Trailing stop increment.

        Returns:
            Deal confirmation dictionary.
        """
        if self.trading_mode == "live":
            return self._open_live_position(epic, direction, size, stop_loss, take_profit)
        return self._open_paper_position(epic, direction, size, stop_loss, take_profit, trailing_stop)

    @staticmethod
    def _price_precision(epic: str) -> int:
        """Return the decimal precision for an instrument's price.

        OANDA requires JPY pairs to use 3 decimal places,
        and most other pairs to use 5.
        """
        return 3 if "JPY" in epic else 5

    @staticmethod
    def _format_price(price: float, epic: str) -> str:
        """Format a price to the correct precision for OANDA."""
        precision = 3 if "JPY" in epic else 5
        return f"{price:.{precision}f}"

    def _lots_to_units(self, epic: str, size: float) -> int:
        """Convert internal lot size to OANDA units.

        JPY pairs: 1 lot = 100 units
        Other pairs: 1 lot = 10,000 units
        """
        if "JPY" in epic:
            return int(round(size * 100))
        return int(round(size * 10_000))

    def _open_live_position(
        self,
        epic: str,
        direction: str,
        size: float,
        stop_loss: float | None,
        take_profit: float | None,
    ) -> dict[str, Any]:
        """Place a real order via OANDA v20 API."""
        try:
            units = self._lots_to_units(epic, size)
            if direction == "SELL":
                units = -units

            order: dict[str, Any] = {
                "type": "MARKET",
                "instrument": epic,
                "units": str(units),
            }
            if stop_loss is not None:
                order["stopLossOnFill"] = {
                    "price": self._format_price(stop_loss, epic),
                    "timeInForce": "GTC",
                }
            if take_profit is not None:
                order["takeProfitOnFill"] = {
                    "price": self._format_price(take_profit, epic),
                    "timeInForce": "GTC",
                }

            resp = self._request(
                "POST",
                f"/v3/accounts/{self.account_id}/orders",
                json_body={"order": order},
            )
            if not resp:
                return {"success": False, "reason": "No response from OANDA"}

            fill = resp.get("orderFillTransaction") or resp.get("relatedTransactionIDs")
            trade_opened = (resp.get("orderFillTransaction") or {}).get("tradeOpened", {})
            trade_id = trade_opened.get("tradeID", "")
            entry_price = float((resp.get("orderFillTransaction") or {}).get("price", 0))

            if not trade_id:
                reason = (resp.get("orderRejectTransaction") or {}).get("rejectReason", "Unknown")
                logger.error(f"[LIVE] Order rejected for {epic}: {reason}")
                return {"success": False, "reason": reason}

            deal_id = f"OANDA-{trade_id}"
            logger.info(
                f"[LIVE] Opened {direction} {epic}: "
                f"units={units}, entry={entry_price:.5f}, "
                f"SL={stop_loss}, TP={take_profit}, trade_id={trade_id}"
            )
            return {
                "success": True,
                "deal_id": deal_id,
                "deal_ref": deal_id,
                "status": "ACCEPTED",
                "direction": direction,
                "epic": epic,
                "size": size,
                "level": entry_price,
                "stop_level": stop_loss or 0,
                "limit_level": take_profit or 0,
            }
        except Exception as e:
            logger.error(f"Failed to open live position for {epic}: {e}")
            return {"success": False, "reason": str(e)}

    def _open_paper_position(
        self,
        epic: str,
        direction: str,
        size: float,
        stop_loss: float | None,
        take_profit: float | None,
        trailing_stop: bool,
    ) -> dict[str, Any]:
        """Simulate a position locally without calling OANDA orders API."""
        try:
            price_info = self.get_market_info(epic)
            if not price_info:
                return {"success": False, "reason": "Could not get current price"}

            entry_price = (
                price_info["offer"] if direction == "BUY" else price_info["bid"]
            )

            deal_id = f"PAPER-{self._next_deal_id:06d}"
            self._next_deal_id += 1

            self._paper_positions[epic] = {
                "deal_id": deal_id,
                "direction": direction,
                "entry_price": entry_price,
                "size": size,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "trailing_stop": trailing_stop,
                "opened_at": datetime.now().isoformat(),
            }

            logger.info(
                f"[PAPER] Opened {direction} {epic}: "
                f"size={size}, entry={entry_price:.5f}, "
                f"SL={stop_loss}, TP={take_profit}"
            )
            return {
                "success": True,
                "deal_id": deal_id,
                "deal_ref": deal_id,
                "status": "ACCEPTED",
                "direction": direction,
                "epic": epic,
                "size": size,
                "level": entry_price,
                "stop_level": stop_loss or 0,
                "limit_level": take_profit or 0,
            }
        except Exception as e:
            logger.error(f"Failed to open paper position: {e}")
            return {"success": False, "reason": str(e)}

    def close_position(
        self,
        deal_id: str,
        direction: str,
        size: float,
        order_type: str = "MARKET",
    ) -> dict[str, Any]:
        """Close a position (paper or live).

        Args:
            deal_id: The deal ID (OANDA-{tradeID} for live, PAPER-XXXXXX for paper).
            direction: Opposite direction.
            size: Position size to close.
            order_type: Order type.

        Returns:
            Close confirmation dictionary.
        """
        if self.trading_mode == "live" and deal_id.startswith("OANDA-"):
            return self._close_live_position(deal_id)
        return self._close_paper_position(deal_id)

    def _close_live_position(self, deal_id: str) -> dict[str, Any]:
        """Close a live OANDA trade."""
        try:
            trade_id = deal_id.removeprefix("OANDA-")
            resp = self._request(
                "PUT",
                f"/v3/accounts/{self.account_id}/trades/{trade_id}/close",
            )
            if resp and "orderFillTransaction" in resp:
                logger.info(f"[LIVE] Closed trade: {deal_id}")
                return {"success": True, "deal_id": deal_id, "status": "ACCEPTED"}
            reason = str(resp) if resp else "No response"
            logger.error(f"[LIVE] Failed to close trade {deal_id}: {reason}")
            return {"success": False, "reason": reason}
        except Exception as e:
            logger.error(f"Failed to close live position {deal_id}: {e}")
            return {"success": False, "reason": str(e)}

    def _close_paper_position(self, deal_id: str) -> dict[str, Any]:
        """Remove a paper position from local state."""
        try:
            for epic, pos in list(self._paper_positions.items()):
                if pos["deal_id"] == deal_id:
                    del self._paper_positions[epic]
                    logger.info(f"[PAPER] Closed position: {deal_id}")
                    return {"success": True, "deal_id": deal_id, "status": "ACCEPTED"}
            return {"success": False, "reason": "Position not found"}
        except Exception as e:
            logger.error(f"Failed to close paper position {deal_id}: {e}")
            return {"success": False, "reason": str(e)}

    def update_position(
        self,
        deal_id: str,
        stop_level: float | None = None,
        limit_level: float | None = None,
        trailing_stop: bool = False,
        trailing_stop_distance: float | None = None,
        trailing_stop_increment: float | None = None,
        epic: str = "",
    ) -> dict[str, Any]:
        """Update stop-loss and take-profit (paper or live).

        Args:
            deal_id: Deal ID to update.
            stop_level: New stop-loss price level.
            limit_level: New take-profit price level.
            trailing_stop: Enable trailing stop.
            trailing_stop_distance: Trailing stop distance.
            trailing_stop_increment: Trailing stop increment.
            epic: Instrument name (for price precision).

        Returns:
            Update confirmation dictionary.
        """
        if self.trading_mode == "live" and deal_id.startswith("OANDA-"):
            return self._update_live_position(deal_id, stop_level, limit_level, epic)
        return self._update_paper_position(deal_id, stop_level, limit_level)

    def _update_live_position(
        self,
        deal_id: str,
        stop_level: float | None,
        limit_level: float | None,
        epic: str = "",
    ) -> dict[str, Any]:
        """Update SL/TP for a live OANDA trade."""
        try:
            trade_id = deal_id.removeprefix("OANDA-")
            body: dict[str, Any] = {}
            if stop_level is not None:
                body["stopLoss"] = {"price": self._format_price(stop_level, epic), "timeInForce": "GTC"}
            if limit_level is not None:
                body["takeProfit"] = {"price": self._format_price(limit_level, epic), "timeInForce": "GTC"}
            if not body:
                return {"success": True, "deal_id": deal_id}

            resp = self._request(
                "PUT",
                f"/v3/accounts/{self.account_id}/trades/{trade_id}/orders",
                json_body=body,
            )
            if resp:
                logger.info(f"[LIVE] Updated trade {deal_id}: SL={stop_level}, TP={limit_level}")
                return {"success": True, "deal_id": deal_id}
            return {"success": False, "reason": "No response from OANDA"}
        except Exception as e:
            logger.error(f"Failed to update live position {deal_id}: {e}")
            return {"success": False, "reason": str(e)}

    def _update_paper_position(
        self,
        deal_id: str,
        stop_level: float | None,
        limit_level: float | None,
    ) -> dict[str, Any]:
        """Update SL/TP for a paper position."""
        try:
            for epic, pos in self._paper_positions.items():
                if pos["deal_id"] == deal_id:
                    if stop_level is not None:
                        pos["stop_loss"] = stop_level
                    if limit_level is not None:
                        pos["take_profit"] = limit_level
                    logger.info(
                        f"[PAPER] Updated position {deal_id}: "
                        f"SL={stop_level}, TP={limit_level}"
                    )
                    return {"success": True, "deal_id": deal_id}
            return {"success": False, "reason": "Position not found"}
        except Exception as e:
            logger.error(f"Failed to update paper position {deal_id}: {e}")
            return {"success": False, "reason": str(e)}

    # --------------------------------------------------
    # Market Sentiment (not available in OANDA free tier)
    # --------------------------------------------------

    def get_client_sentiment(self, market_id: str) -> dict[str, Any]:
        """Get client sentiment data (placeholder).

        Args:
            market_id: Market ID.

        Returns:
            Default neutral sentiment.
        """
        return {"long_pct": 50, "short_pct": 50, "market_id": market_id}

    # --------------------------------------------------
    # Internal Helpers
    # --------------------------------------------------

    def stream_prices(
        self,
        instruments: list[str],
        on_price: Any,
        stop_event: Any | None = None,
    ) -> None:
        """OANDAストリーミングAPIで価格をリアルタイム受信する。

        Args:
            instruments: 受信する通貨ペアのリスト（例: ['USD_JPY', 'EUR_USD']）
            on_price: 価格更新コールバック on_price(instrument, bid, ask, time)
            stop_event: threading.Event。セットされたら停止する。
        """
        stream_url = (
            "https://stream-fxpractice.oanda.com"
            if self.environment == "practice"
            else "https://stream-fxtrade.oanda.com"
        )
        url = f"{stream_url}/v3/accounts/{self.account_id}/pricing/stream"
        params = {"instruments": ",".join(instruments)}
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Accept-Datetime-Format": "RFC3339",
        }
        import json as _json
        while True:
            if stop_event and stop_event.is_set():
                break
            try:
                with requests.get(
                    url, params=params, headers=headers,
                    stream=True, timeout=30
                ) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines():
                        if stop_event and stop_event.is_set():
                            return
                        if not line:
                            continue
                        try:
                            msg = _json.loads(line)
                        except Exception:
                            continue
                        if msg.get("type") != "PRICE":
                            continue
                        instrument = msg.get("instrument", "")
                        bids = msg.get("bids", [{}])
                        asks = msg.get("asks", [{}])
                        bid = float(bids[0].get("price", 0)) if bids else 0.0
                        ask = float(asks[0].get("price", 0)) if asks else 0.0
                        ts = msg.get("time", "")
                        on_price(instrument, bid, ask, ts)
            except Exception as e:
                logger.warning(f"Price stream disconnected: {e} — reconnecting in 3s")
                if stop_event and stop_event.is_set():
                    break
                time.sleep(3)

    def _request(
        self,
        method: str,
        path: str,
        params: dict | None = None,
        json_body: dict | None = None,
    ) -> dict[str, Any] | None:
        """Make an authenticated API request.

        Args:
            method: HTTP method.
            path: API path.
            params: Query parameters.
            json_body: JSON request body.

        Returns:
            Response JSON or None on error.
        """
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update({
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json",
                "Accept-Datetime-Format": "RFC3339",
            })

        url = f"{self._base_url}{path}"

        try:
            resp = self._session.request(
                method, url, params=params, json=json_body, timeout=30
            )

            if resp.status_code == 429:
                # Rate limited — wait and retry once
                retry_after = int(resp.headers.get("Retry-After", "1"))
                logger.warning(f"Rate limited, waiting {retry_after}s...")
                time.sleep(retry_after)
                resp = self._session.request(
                    method, url, params=params, json=json_body, timeout=30
                )

            resp.raise_for_status()
            return resp.json()

        except requests.exceptions.HTTPError as e:
            logger.error(f"OANDA API error: {e} - {resp.text if resp else ''}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"OANDA request failed: {e}")
            return None

    @staticmethod
    def _candles_to_dataframe(candles: list[dict]) -> pd.DataFrame:
        """Convert OANDA candle list to OHLCV DataFrame.

        Args:
            candles: List of candle dicts from OANDA API.

        Returns:
            DataFrame with OHLCV columns and datetime index.
        """
        rows = []
        for c in candles:
            if not c.get("complete", True):
                continue  # Skip incomplete candles
            mid = c.get("mid", {})
            rows.append({
                "datetime": c["time"],
                "open": float(mid.get("o", 0)),
                "high": float(mid.get("h", 0)),
                "low": float(mid.get("l", 0)),
                "close": float(mid.get("c", 0)),
                "volume": int(c.get("volume", 0)),
            })

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime")
        df.index.name = "datetime"
        return df
