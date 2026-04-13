"""OANDA v20 API Client - Market data and order execution.

Provides a high-level wrapper around the OANDA v20 REST API
for fetching forex market data and executing orders.

Order execution policy:
  environment=practice → All orders sent to OANDA practice account API
  environment=live     → All orders sent to OANDA live account API
  trading_mode=paper   → Legacy local simulation (only when no OANDA env set)

Since the user always connects to an OANDA account (practice or live),
all position and order management uses the OANDA API regardless of trading_mode.
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
        # Always use OANDA API when connected to an OANDA account (practice or live).
        # Local paper simulation is only a last-resort fallback when no account is configured.
        self._use_oanda_api: bool = bool(api_token and account_id)
        self._session: requests.Session | None = None
        self._connected = False

        # Paper trading state
        self._paper_positions: dict[str, dict[str, Any]] = {}
        self._paper_balance: float = 1_000_000.0  # Default 100万円
        self._paper_pnl: float = 0.0
        self._next_deal_id: int = 1

        # Instrument info cache (from OANDA API)
        self._instrument_cache: dict[str, dict[str, Any]] = {}
        self._account_currency: str = "JPY"

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
                self._account_currency = account.get("currency", "JPY")
                self._connected = True
                logger.info(
                    f"Connected to OANDA ({self.environment}) "
                    f"account={self.account_id} currency={self._account_currency}"
                )

                # Pre-fetch instrument metadata from OANDA
                self._fetch_instrument_cache()
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
    # Instrument Metadata (from OANDA API)
    # --------------------------------------------------

    def _fetch_instrument_cache(self) -> None:
        """Fetch and cache instrument metadata from OANDA API.

        Caches pipLocation, displayPrecision, minimumTradeSize, marginRate
        for all available instruments.
        """
        try:
            resp = self._request(
                "GET", f"/v3/accounts/{self.account_id}/instruments",
            )
            if resp and "instruments" in resp:
                for inst in resp["instruments"]:
                    name = inst["name"]
                    self._instrument_cache[name] = {
                        "pipLocation": inst.get("pipLocation", -4),
                        "displayPrecision": inst.get("displayPrecision", 5),
                        "minimumTradeSize": int(inst.get("minimumTradeSize", "1")),
                        "maximumOrderUnits": int(inst.get("maximumOrderUnits", "100000000")),
                        "marginRate": float(inst.get("marginRate", "0.04")),
                        "type": inst.get("type", "CURRENCY"),
                    }
                logger.info(
                    f"Instrument cache loaded: {len(self._instrument_cache)} instruments"
                )
        except Exception as e:
            logger.error(f"Failed to fetch instrument cache: {e}")

    def get_instrument_info(self, instrument: str) -> dict[str, Any]:
        """Get cached instrument metadata.

        Returns:
            Dict with pipLocation, displayPrecision, etc.
        """
        if not self._instrument_cache:
            self._fetch_instrument_cache()
        return self._instrument_cache.get(instrument, {})

    def get_pip_location(self, instrument: str) -> int:
        """Get pip location exponent from OANDA (e.g., -4 for 0.0001, -2 for 0.01).

        Args:
            instrument: OANDA instrument name (e.g., 'GBP_USD').

        Returns:
            Pip location exponent. Defaults to -2 for JPY pairs, -4 for others.
        """
        info = self.get_instrument_info(instrument)
        if info:
            return info["pipLocation"]
        return -2 if "JPY" in instrument else -4

    def get_pip_value(self, instrument: str) -> float:
        """Get the pip value (e.g., 0.01 for JPY pairs, 0.0001 for others).

        Derived from OANDA's pipLocation field.
        """
        return 10.0 ** self.get_pip_location(instrument)

    def get_pip_value_in_account_currency(
        self, instrument: str, units: int = 1
    ) -> float:
        """Calculate the value of 1 pip per unit in the account currency (JPY).

        Uses OANDA pricing API for live FX rate conversion.
        For 1 unit: 1 pip move = pip_value (in quote currency)
        If quote currency != account currency, convert via live rate.

        Args:
            instrument: OANDA instrument (e.g., 'GBP_USD').
            units: Number of units (default 1).

        Returns:
            Value of 1 pip in account currency (JPY) for the given units.
        """
        pip_value = self.get_pip_value(instrument)
        # Determine quote currency (right side of pair)
        parts = instrument.split("_")
        quote_currency = parts[1] if len(parts) == 2 else ""

        if quote_currency == self._account_currency:
            # e.g., USD_JPY — pip is already in JPY
            return pip_value * units

        # Need to convert quote currency to account currency
        # e.g., GBP_USD → quote=USD, need USD→JPY rate
        conversion_rate = self._get_conversion_rate(quote_currency, self._account_currency)
        return pip_value * units * conversion_rate

    def _get_conversion_rate(self, from_currency: str, to_currency: str) -> float:
        """Get live FX rate to convert from one currency to another.

        Tries {from}_{to} first, then 1/{to}_{from}.

        Args:
            from_currency: Source currency (e.g., 'USD').
            to_currency: Target currency (e.g., 'JPY').

        Returns:
            Conversion rate. Returns 1.0 on failure.
        """
        if from_currency == to_currency:
            return 1.0

        # Try direct pair
        direct_pair = f"{from_currency}_{to_currency}"
        rate = self._get_mid_price(direct_pair)
        if rate > 0:
            return rate

        # Try inverse pair
        inverse_pair = f"{to_currency}_{from_currency}"
        rate = self._get_mid_price(inverse_pair)
        if rate > 0:
            return 1.0 / rate

        logger.warning(f"Could not get conversion rate {from_currency}→{to_currency}")
        return 1.0

    def _get_mid_price(self, instrument: str) -> float:
        """Get the mid price for an instrument from OANDA pricing API."""
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
                if bid > 0 and ask > 0:
                    return (bid + ask) / 2
        except Exception:
            pass
        return 0.0

    def calculate_units(
        self,
        instrument: str,
        risk_amount_account: float,
        stop_distance_price: float,
    ) -> int:
        """Calculate the number of OANDA units for a given risk amount.

        This is the CORRECT way to size positions:
        units = risk_amount / (stop_distance * pip_value_per_unit_in_account_currency)

        Args:
            instrument: OANDA instrument (e.g., 'GBP_USD').
            risk_amount_account: Max loss in account currency (JPY).
            stop_distance_price: Absolute price distance to stop loss.

        Returns:
            Number of units (always positive).
        """
        pip_value = self.get_pip_value(instrument)
        if pip_value <= 0 or stop_distance_price <= 0:
            return 0

        stop_pips = stop_distance_price / pip_value
        pip_value_per_unit = self.get_pip_value_in_account_currency(instrument, units=1)

        if pip_value_per_unit <= 0 or stop_pips <= 0:
            return 0

        units = risk_amount_account / (stop_pips * pip_value_per_unit)

        # Respect OANDA maximum
        info = self.get_instrument_info(instrument)
        max_units = info.get("maximumOrderUnits", 100_000_000)
        units = min(units, max_units)

        return max(int(units), 1)

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
        if self._use_oanda_api:
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

                sl_order = trade.get("stopLossOrder", {})
                tp_order = trade.get("takeProfitOrder", {})
                rows.append({
                    "dealId": f"OANDA-{trade['id']}",
                    "epic": instrument,
                    "direction": direction,
                    "size": size_units,  # OANDA units (not lots)
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
        if self._use_oanda_api:
            return self._open_live_position(epic, direction, size, stop_loss, take_profit)
        return self._open_paper_position(epic, direction, size, stop_loss, take_profit, trailing_stop)

    def _price_precision(self, epic: str) -> int:
        """Return the decimal precision for an instrument's price.

        Uses OANDA API's displayPrecision field from instrument cache.
        """
        info = self.get_instrument_info(epic)
        if info:
            return info["displayPrecision"]
        return 3 if "JPY" in epic else 5  # last-resort fallback

    def _format_price(self, price: float, epic: str) -> str:
        """Format a price to the correct precision for OANDA."""
        precision = self._price_precision(epic)
        return f"{price:.{precision}f}"

    def _open_live_position(
        self,
        epic: str,
        direction: str,
        size: float,
        stop_loss: float | None,
        take_profit: float | None,
    ) -> dict[str, Any]:
        """Place a real order via OANDA v20 API.

        Args:
            size: Number of OANDA units (NOT lots).
        """
        try:
            units = int(round(size))
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
            tag = "PRACTICE" if self.environment == "practice" else "LIVE"
            logger.info(
                f"[{tag}] Opened {direction} {epic}: "
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
        if self._use_oanda_api:
            return self._close_live_position(deal_id)
        return self._close_paper_position(deal_id)

    def _close_live_position(self, deal_id: str) -> dict[str, Any]:
        """Close an OANDA trade (practice or live account)."""
        try:
            trade_id = deal_id.removeprefix("OANDA-")
            resp = self._request(
                "PUT",
                f"/v3/accounts/{self.account_id}/trades/{trade_id}/close",
            )
            if resp and "orderFillTransaction" in resp:
                fill = resp["orderFillTransaction"]
                realized_pl = float(fill.get("pl", 0))
                exit_price = float(fill.get("price", 0))
                logger.info(
                    f"[OANDA] Closed trade: {deal_id} "
                    f"exit={exit_price} pl={realized_pl:+.2f}"
                )
                return {
                    "success": True,
                    "deal_id": deal_id,
                    "status": "ACCEPTED",
                    "realized_pl": realized_pl,
                    "exit_price": exit_price,
                }
            reason = str(resp) if resp else "No response"
            logger.error(f"[OANDA] Failed to close trade {deal_id}: {reason}")
            return {"success": False, "reason": reason}
        except Exception as e:
            logger.error(f"Failed to close OANDA position {deal_id}: {e}")
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
        if self._use_oanda_api:
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
    # OANDA API - Source of Truth for Position Data
    # --------------------------------------------------

    def get_positions(self) -> list[dict[str, Any]]:
        """Get all open positions as a list of dicts.

        OANDA API is the single source of truth for position data.
        Only AI metadata (pattern, confidence, signal_id) is stored locally.

        Returns:
            List of position dicts with deal_id, instrument, direction, size, etc.
        """
        if self._use_oanda_api:
            try:
                resp = self._request("GET", f"/v3/accounts/{self.account_id}/openTrades")
                if not resp or "trades" not in resp:
                    return []
                result = []
                for trade in resp["trades"]:
                    units = float(trade.get("currentUnits", 0))
                    sl_order = trade.get("stopLossOrder", {})
                    tp_order = trade.get("takeProfitOrder", {})
                    result.append({
                        "deal_id": f"OANDA-{trade['id']}",
                        "instrument": trade.get("instrument", ""),
                        "direction": "BUY" if units > 0 else "SELL",
                        "size": abs(units),
                        "entry_price": float(trade.get("price", 0)),
                        "stop_loss": float(sl_order.get("price", 0) or 0) if sl_order else 0.0,
                        "take_profit": float(tp_order.get("price", 0) or 0) if tp_order else 0.0,
                        "unrealized_pl": float(trade.get("unrealizedPL", 0)),
                        "open_time": trade.get("openTime", ""),
                    })
                return result
            except Exception as e:
                logger.error(f"Failed to get positions from OANDA: {e}")
                return []

        # Paper mode: convert _paper_positions dict to list
        return [
            {
                "deal_id": pos["deal_id"],
                "instrument": epic,
                "direction": pos["direction"],
                "size": pos["size"],
                "entry_price": pos["entry_price"],
                "stop_loss": pos.get("stop_loss", 0) or 0.0,
                "take_profit": pos.get("take_profit", 0) or 0.0,
                "unrealized_pl": 0.0,
                "open_time": pos.get("opened_at", ""),
            }
            for epic, pos in self._paper_positions.items()
        ]

    def get_closed_trade(self, trade_id: str) -> dict[str, Any] | None:
        """Fetch a single closed trade's details from OANDA API.

        Used to get the actual realizedPL for a closed position.

        Args:
            trade_id: Deal ID (e.g., 'OANDA-123' or raw trade ID '123').

        Returns:
            Dict with realized_pl, exit_price, close_time, or None on failure.
        """
        if not self._use_oanda_api:
            return None
        try:
            raw_id = trade_id.removeprefix("OANDA-")
            resp = self._request("GET", f"/v3/accounts/{self.account_id}/trades/{raw_id}")
            if not resp or "trade" not in resp:
                return None
            trade = resp["trade"]
            return {
                "trade_id": trade_id,
                "realized_pl": float(trade.get("realizedPL", 0)),
                "exit_price": float(trade.get("averageClosePrice", 0) or 0),
                "close_time": trade.get("closeTime", ""),
            }
        except Exception as e:
            logger.error(f"Failed to get closed trade {trade_id}: {e}")
            return None

    def get_closed_trades(self, count: int = 50) -> list[dict[str, Any]]:
        """Fetch recent closed trades from OANDA API.

        Args:
            count: Number of trades to fetch (max 500).

        Returns:
            List of closed trade dicts with realized_pl, exit_price, etc.
        """
        if not self._use_oanda_api:
            return []
        try:
            resp = self._request(
                "GET",
                f"/v3/accounts/{self.account_id}/trades",
                params={"state": "CLOSED", "count": min(count, 500)},
            )
            if not resp or "trades" not in resp:
                return []
            return [
                {
                    "trade_id": f"OANDA-{t['id']}",
                    "instrument": t.get("instrument", ""),
                    "realized_pl": float(t.get("realizedPL", 0)),
                    "exit_price": float(t.get("averageClosePrice", 0) or 0),
                    "close_time": t.get("closeTime", ""),
                }
                for t in resp["trades"]
            ]
        except Exception as e:
            logger.error(f"Failed to get closed trades from OANDA: {e}")
            return []

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
        max_retries = 3
        last_err: Exception | None = None

        for attempt in range(1, max_retries + 1):
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
            except requests.exceptions.ConnectionError as e:
                # DNS failure, connection reset, etc. — retry with backoff
                last_err = e
                if attempt < max_retries:
                    wait = 2 ** attempt  # 2s, 4s
                    logger.warning(
                        f"OANDA connection error (attempt {attempt}/{max_retries}), "
                        f"retrying in {wait}s: {e}"
                    )
                    time.sleep(wait)
                    # Re-create session in case connection pool is stale
                    self._session = requests.Session()
                    self._session.headers.update({
                        "Authorization": f"Bearer {self.api_token}",
                        "Content-Type": "application/json",
                        "Accept-Datetime-Format": "RFC3339",
                    })
            except requests.exceptions.RequestException as e:
                logger.error(f"OANDA request failed: {e}")
                return None

        logger.error(f"OANDA request failed after {max_retries} retries: {last_err}")
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
