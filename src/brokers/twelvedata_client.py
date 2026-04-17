"""Twelve Data API Client - Free forex market data.

Provides the same interface as OandaClient for seamless switching.
Uses the Twelve Data REST API for historical and real-time forex data.
Paper trading is simulated locally.

Free tier: 800 API calls/day, 8 calls/minute.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

import pandas as pd
import requests
from loguru import logger


# Map IG-style / OANDA-style resolution names to Twelve Data intervals
_INTERVAL_MAP = {
    # IG-style
    "MINUTE": "1min",
    "MINUTE_5": "5min",
    "MINUTE_15": "15min",
    "MINUTE_30": "30min",
    "HOUR": "1h",
    "HOUR_2": "2h",
    "HOUR_4": "4h",
    "DAY": "1day",
    "WEEK": "1week",
    "MONTH": "1month",
    # OANDA-style
    "M1": "1min",
    "M5": "5min",
    "M15": "15min",
    "M30": "30min",
    "H1": "1h",
    "H2": "2h",
    "H4": "4h",
    "D": "1day",
    "W": "1week",
    "M": "1month",
    # Twelve Data native (pass through)
    "1min": "1min",
    "5min": "5min",
    "15min": "15min",
    "30min": "30min",
    "45min": "45min",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "1day": "1day",
    "1week": "1week",
    "1month": "1month",
}

# Map OANDA-style instrument names to Twelve Data symbols
# OANDA: "USD_JPY" -> Twelve Data: "USD/JPY"
def _to_twelvedata_symbol(instrument: str) -> str:
    """Convert instrument name to Twelve Data symbol format."""
    if "/" in instrument:
        return instrument
    if "_" in instrument:
        return instrument.replace("_", "/")
    return instrument


class TwelveDataClient:
    """Client for Twelve Data REST API.

    Provides the same interface as OandaClient so the two can be
    swapped via configuration alone.
    """

    BASE_URL = "https://api.twelvedata.com"

    def __init__(
        self,
        api_key: str,
        **kwargs: Any,
    ) -> None:
        """Initialize Twelve Data client.

        Args:
            api_key: Twelve Data API key.
            **kwargs: Ignored (accepts OandaClient params for compatibility).
        """
        self.api_key = api_key
        self._session: requests.Session | None = None
        self._connected = False
        self._last_request_time: float = 0
        self._min_request_interval = 8.0  # 8 calls/min = ~7.5s between calls

        # Paper trading state
        self._paper_positions: dict[str, dict[str, Any]] = {}  # key=deal_id
        self._paper_balance: float = 1_000_000.0
        self._paper_realized_pnl: float = 0.0   # 確定済み累積損益
        self._next_deal_id: int = 1

    # --------------------------------------------------
    # Session Management
    # --------------------------------------------------

    def connect(self) -> bool:
        """Validate API key by making a test request.

        Returns:
            True if API key is valid.
        """
        try:
            self._session = requests.Session()

            # Test with a simple request
            resp = self._request("/time_series", {
                "symbol": "USD/JPY",
                "interval": "1day",
                "outputsize": 1,
            })

            if resp and "values" in resp:
                self._connected = True
                logger.info("Connected to Twelve Data API")
                return True

            if resp and resp.get("code") == 401:
                logger.error("Twelve Data: invalid API key")
            else:
                logger.error(f"Twelve Data connection failed: {resp}")
            return False

        except Exception as e:
            logger.error(f"Twelve Data connection error: {e}")
            return False

    def disconnect(self) -> None:
        """Close the session."""
        if self._session:
            self._session.close()
            self._session = None
        self._connected = False
        logger.info("Disconnected from Twelve Data")

    def ensure_session(self) -> None:
        """Ensure the session is active."""
        if not self._connected or self._session is None:
            self.connect()

    # --------------------------------------------------
    # Account Information (Paper Trading)
    # --------------------------------------------------

    def get_account_info(self) -> dict[str, Any]:
        """Get paper account info.

        Returns:
            Account information dictionary.
        """
        return {
            "account_id": "TWELVEDATA-PAPER",
            "account_name": "Twelve Data (Paper)",
            "balance": self._paper_balance,
            "deposit": 1_000_000.0,
            "profit_loss": self._paper_realized_pnl,
            "available": self._paper_balance,
            "currency": "JPY",
        }

    def apply_trade_pnl(self, pnl: float) -> None:
        """Apply realized P&L to paper balance after a position closes.

        Args:
            pnl: Profit/loss amount in account currency (JPY).
        """
        self._paper_balance += pnl
        self._paper_realized_pnl += pnl
        logger.info(
            f"[PAPER] Balance updated: {pnl:+,.0f} → "
            f"残高 ¥{self._paper_balance:,.0f} "
            f"(累積損益 {self._paper_realized_pnl:+,.0f})"
        )

    # --------------------------------------------------
    # Market Data
    # --------------------------------------------------

    def search_markets(self, term: str) -> pd.DataFrame:
        """Search for forex symbols.

        Args:
            term: Search term (e.g., 'USD/JPY', 'EUR').

        Returns:
            DataFrame of matching instruments.
        """
        try:
            resp = self._request("/symbol_search", {
                "symbol": term,
                "show_plan": "false",
            })
            if resp and "data" in resp:
                matches = [
                    {
                        "instrument": item.get("symbol", ""),
                        "displayName": item.get("instrument_name", ""),
                        "type": item.get("instrument_type", ""),
                        "exchange": item.get("exchange", ""),
                    }
                    for item in resp["data"]
                    if item.get("instrument_type") == "Physical Currency"
                ]
                return pd.DataFrame(matches)
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Market search failed for '{term}': {e}")
            return pd.DataFrame()

    def get_market_info(self, instrument: str) -> dict[str, Any]:
        """Get current price for an instrument.

        Args:
            instrument: Instrument name (e.g., 'USD_JPY').

        Returns:
            Market details dictionary.
        """
        symbol = _to_twelvedata_symbol(instrument)
        try:
            resp = self._request("/price", {"symbol": symbol})
            if resp and "price" in resp:
                price = float(resp["price"])
                # Twelve Data returns a single mid price;
                # simulate a small spread
                spread = 0.02 if "JPY" in symbol else 0.00020
                bid = price - spread / 2
                ask = price + spread / 2
                return {
                    "epic": instrument,
                    "name": symbol,
                    "type": "CURRENCIES",
                    "bid": bid,
                    "offer": ask,
                    "high": 0,
                    "low": 0,
                    "spread": spread,
                    "market_status": "TRADEABLE",
                    "min_deal_size": 1,
                    "lot_size": 1,
                    "currency": "JPY" if "JPY" in symbol else "USD",
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
            instrument: Instrument name (e.g., 'USD_JPY').
            resolution: Candle resolution (e.g., 'HOUR_4', 'DAY', '4h').
            num_points: Number of candles to fetch (max 5000).

        Returns:
            DataFrame with OHLCV data.
        """
        symbol = _to_twelvedata_symbol(instrument)
        interval = _INTERVAL_MAP.get(resolution, resolution)
        if interval not in _INTERVAL_MAP.values():
            logger.error(f"Invalid resolution: {resolution}")
            return pd.DataFrame()

        try:
            resp = self._request("/time_series", {
                "symbol": symbol,
                "interval": interval,
                "outputsize": min(num_points, 5000),
                "format": "JSON",
            })

            if resp and "values" in resp:
                return self._values_to_dataframe(resp["values"])

            if resp and "message" in resp:
                logger.error(f"Twelve Data error: {resp['message']}")
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
            instrument: Instrument name.
            resolution: Candle resolution.
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).

        Returns:
            DataFrame with OHLCV data.
        """
        symbol = _to_twelvedata_symbol(instrument)
        interval = _INTERVAL_MAP.get(resolution, resolution)
        if interval not in _INTERVAL_MAP.values():
            logger.error(f"Invalid resolution: {resolution}")
            return pd.DataFrame()

        try:
            resp = self._request("/time_series", {
                "symbol": symbol,
                "interval": interval,
                "start_date": start_date,
                "end_date": end_date,
                "format": "JSON",
            })

            if resp and "values" in resp:
                return self._values_to_dataframe(resp["values"])

            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Failed to fetch date-range prices for {instrument}: {e}")
            return pd.DataFrame()

    # --------------------------------------------------
    # Position Management (Paper Trading)
    # --------------------------------------------------

    def get_open_positions(self) -> pd.DataFrame:
        """Fetch all open paper positions."""
        if not self._paper_positions:
            return pd.DataFrame()

        rows = []
        for deal_id, pos in self._paper_positions.items():
            rows.append({
                "dealId": deal_id,
                "epic": pos["epic"],
                "direction": pos["direction"],
                "size": pos["size"],
                "level": pos["entry_price"],
                "stopLevel": pos.get("stop_loss"),
                "limitLevel": pos.get("take_profit"),
                "openedAt": pos.get("opened_at", ""),
            })
        return pd.DataFrame(rows)

    def get_working_orders(self) -> pd.DataFrame:
        """Not supported in paper mode."""
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
        """Open a paper position."""
        try:
            price_info = self.get_market_info(epic)
            if not price_info:
                return {"success": False, "reason": "Could not get current price"}

            entry_price = (
                price_info["offer"] if direction == "BUY" else price_info["bid"]
            )

            deal_id = f"PAPER-{self._next_deal_id:06d}"
            self._next_deal_id += 1

            self._paper_positions[deal_id] = {
                "epic": epic,
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
        """Close a paper position."""
        try:
            if deal_id in self._paper_positions:
                del self._paper_positions[deal_id]
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
    ) -> dict[str, Any]:
        """Update a paper position."""
        try:
            if deal_id in self._paper_positions:
                pos = self._paper_positions[deal_id]
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

    def get_client_sentiment(self, market_id: str) -> dict[str, Any]:
        """Sentiment not available — return neutral."""
        return {"long_pct": 50, "short_pct": 50, "market_id": market_id}

    # --------------------------------------------------
    # Internal Helpers
    # --------------------------------------------------

    def _request(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
        _retry: int = 0,
    ) -> dict[str, Any] | None:
        """Make a rate-limited API request with exponential backoff retry.

        Args:
            endpoint: API endpoint path.
            params: Query parameters.
            _retry: Current retry count (internal use).

        Returns:
            Response JSON or None.
        """
        max_retries = 4
        base_wait = 5.0  # 初回リトライ待機秒数

        if self._session is None:
            self._session = requests.Session()

        # Rate limiting: enforce minimum interval between requests
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self._min_request_interval:
            wait = self._min_request_interval - elapsed
            logger.debug(f"Rate limit: waiting {wait:.1f}s")
            time.sleep(wait)

        url = f"{self.BASE_URL}{endpoint}"
        params = params or {}
        params["apikey"] = self.api_key

        try:
            resp = self._session.get(url, params=params, timeout=30)
            self._last_request_time = time.time()

            resp.raise_for_status()
            data = resp.json()

            # Twelve Data returns errors inside 200 responses
            if data.get("status") == "error":
                code = data.get("code", 0)
                if code == 429:
                    wait = 60
                    logger.warning(f"Rate limited by Twelve Data, waiting {wait}s...")
                    time.sleep(wait)
                    return self._request(endpoint, params, _retry)
                logger.error(f"Twelve Data error: {data.get('message', data)}")
                return data

            return data

        except requests.exceptions.ConnectionError as e:
            if _retry < max_retries:
                wait = base_wait * (2 ** _retry)
                logger.warning(f"Connection error, retry {_retry + 1}/{max_retries} in {wait:.0f}s: {e}")
                time.sleep(wait)
                self._session = requests.Session()  # セッションをリセット
                return self._request(endpoint, params, _retry + 1)
            logger.error(f"Twelve Data connection failed after {max_retries} retries: {e}")
            return None
        except requests.exceptions.Timeout as e:
            if _retry < max_retries:
                wait = base_wait * (2 ** _retry)
                logger.warning(f"Timeout, retry {_retry + 1}/{max_retries} in {wait:.0f}s")
                time.sleep(wait)
                return self._request(endpoint, params, _retry + 1)
            logger.error(f"Twelve Data request timed out after {max_retries} retries: {e}")
            return None
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else 0
            if status in (500, 502, 503, 504) and _retry < max_retries:
                wait = base_wait * (2 ** _retry)
                logger.warning(f"Server error {status}, retry {_retry + 1}/{max_retries} in {wait:.0f}s")
                time.sleep(wait)
                return self._request(endpoint, params, _retry + 1)
            logger.error(f"Twelve Data HTTP error: {e}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Twelve Data request failed: {e}")
            return None

    @staticmethod
    def _values_to_dataframe(values: list[dict]) -> pd.DataFrame:
        """Convert Twelve Data values list to OHLCV DataFrame.

        Args:
            values: List of candle dicts from Twelve Data.

        Returns:
            DataFrame with OHLCV columns and datetime index.
        """
        if not values:
            return pd.DataFrame()

        rows = []
        for v in values:
            rows.append({
                "datetime": v["datetime"],
                "open": float(v["open"]),
                "high": float(v["high"]),
                "low": float(v["low"]),
                "close": float(v["close"]),
                "volume": int(v.get("volume", 0)) if v.get("volume") else 0,
            })

        df = pd.DataFrame(rows)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)
        df = df.set_index("datetime")
        df.index.name = "datetime"
        return df
