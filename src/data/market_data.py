"""Market data collection and management module.

Fetches, stores, and manages OHLCV data from OANDA
for multiple timeframes and currency pairs.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger


class MarketDataManager:
    """Manage market data collection and caching for FX pairs."""

    def __init__(self, broker_client: Any, data_dir: str = "data") -> None:
        """Initialize MarketDataManager.

        Args:
            broker_client: OandaClient instance for data retrieval.
            data_dir: Directory for data caching.
        """
        self.broker_client = broker_client
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "raw").mkdir(exist_ok=True)
        (self.data_dir / "processed").mkdir(exist_ok=True)

        # In-memory cache
        self._cache: dict[str, pd.DataFrame] = {}

    def fetch_prices(
        self,
        instrument: str,
        resolution: str = "HOUR_4",
        num_points: int = 500,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Fetch OHLCV price data for a market.

        Args:
            instrument: OANDA instrument (e.g., 'USD_JPY').
            resolution: Candle resolution.
            num_points: Number of data points.
            use_cache: Whether to use cached data.

        Returns:
            DataFrame with OHLCV data.
        """
        cache_key = f"{instrument}_{resolution}_{num_points}"

        if use_cache and cache_key in self._cache:
            cached = self._cache[cache_key]
            # Return cache if less than 5 minutes old
            if hasattr(cached, '_fetch_time'):
                if (datetime.now() - cached._fetch_time).seconds < 300:
                    return cached

        df = self.broker_client.get_historical_prices(instrument, resolution, num_points)

        if not df.empty:
            df._fetch_time = datetime.now()  # type: ignore
            self._cache[cache_key] = df
            logger.debug(f"Fetched {len(df)} bars for {instrument} ({resolution})")

        return df

    def fetch_multi_timeframe(
        self,
        instrument: str,
        timeframes: dict[str, str],
        num_points: int = 500,
    ) -> dict[str, pd.DataFrame]:
        """Fetch data across multiple timeframes.

        Args:
            instrument: OANDA instrument.
            timeframes: Dict of name->resolution (e.g., {'primary': 'HOUR_4'}).
            num_points: Data points per timeframe.

        Returns:
            Dictionary of name->DataFrame.
        """
        result = {}
        for name, resolution in timeframes.items():
            df = self.fetch_prices(instrument, resolution, num_points)
            if not df.empty:
                result[name] = df
            else:
                logger.warning(f"No data for {instrument} at {resolution}")
        return result

    def save_to_csv(self, df: pd.DataFrame, filename: str) -> str:
        """Save DataFrame to CSV in data directory.

        Args:
            df: Price data DataFrame.
            filename: Output filename.

        Returns:
            Full path to saved file.
        """
        path = self.data_dir / "raw" / filename
        df.to_csv(str(path))
        logger.info(f"Saved data to {path}")
        return str(path)

    def load_from_csv(self, filename: str) -> pd.DataFrame:
        """Load price data from CSV.

        Args:
            filename: CSV filename.

        Returns:
            DataFrame with OHLCV data.
        """
        path = self.data_dir / "raw" / filename
        if path.exists():
            df = pd.read_csv(str(path), index_col=0, parse_dates=True)
            df.index.name = "datetime"
            return df
        return pd.DataFrame()

    def get_latest_price(self, instrument: str) -> dict[str, float]:
        """Get the latest bid/offer price for a market.

        Args:
            instrument: OANDA instrument.

        Returns:
            Dictionary with bid, offer, mid, spread.
        """
        info = self.broker_client.get_market_info(instrument)
        if info:
            bid = info.get("bid", 0)
            offer = info.get("offer", 0)
            return {
                "bid": bid,
                "offer": offer,
                "mid": (bid + offer) / 2,
                "spread": offer - bid,
            }
        return {"bid": 0, "offer": 0, "mid": 0, "spread": 0}

    def clear_cache(self) -> None:
        """Clear the in-memory data cache."""
        self._cache.clear()
        logger.debug("Data cache cleared")
