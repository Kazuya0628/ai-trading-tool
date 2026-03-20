"""CSV Data Loader for long-term backtesting.

Supports:
1. Downloading multi-year forex data via yfinance (free, unlimited)
2. Importing user-provided CSV files (any OHLCV format)
3. Saving/caching downloaded data for reuse
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

# yfinance symbol mapping: OANDA instrument → Yahoo Finance ticker
YAHOO_SYMBOLS: dict[str, str] = {
    "USD_JPY": "USDJPY=X",
    "EUR_USD": "EURUSD=X",
    "GBP_JPY": "GBPJPY=X",
    "GBP_USD": "GBPUSD=X",
    "EUR_JPY": "EURJPY=X",
    "AUD_USD": "AUDUSD=X",
    "AUD_JPY": "AUDJPY=X",
    "NZD_USD": "NZDUSD=X",
    "USD_CHF": "USDCHF=X",
    "USD_CAD": "USDCAD=X",
    "EUR_GBP": "EURGBP=X",
}

DATA_DIR = Path("data/historical")


def download_forex_data(
    instrument: str,
    years: int = 3,
    interval: str = "4h",
    save: bool = True,
) -> pd.DataFrame:
    """Download historical forex data from Yahoo Finance.

    Args:
        instrument: OANDA-style instrument (e.g., 'USD_JPY').
        years: Number of years of history to download.
        interval: Candle interval ('1h', '4h', '1d').
        save: Whether to save as CSV for future use.

    Returns:
        OHLCV DataFrame with standardized column names.
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance not installed. Run: pip install yfinance")
        return pd.DataFrame()

    symbol = YAHOO_SYMBOLS.get(instrument)
    if not symbol:
        logger.error(f"Unknown instrument: {instrument}. Known: {list(YAHOO_SYMBOLS.keys())}")
        return pd.DataFrame()

    # yfinance interval limits:
    # 1h: max 730 days, 4h: not natively supported (use 1h and resample)
    # 1d: unlimited
    logger.info(f"Downloading {instrument} ({symbol}) - {years}yr @ {interval}...")

    try:
        period = f"{years}y"

        if interval in ("4h", "HOUR_4"):
            # yfinance doesn't support 4h directly — download 1h and resample
            # 1h max is 730 days
            download_period = f"{min(years, 2)}y"
            ticker = yf.Ticker(symbol)
            raw = ticker.history(period=download_period, interval="1h")

            if raw.empty:
                logger.error(f"No data returned for {symbol}")
                return pd.DataFrame()

            df = _resample_to_4h(raw)

        elif interval in ("1d", "DAY"):
            ticker = yf.Ticker(symbol)
            raw = ticker.history(period=period, interval="1d")
            df = _standardize_columns(raw)

        elif interval in ("1h", "HOUR"):
            download_period = f"{min(years, 2)}y"
            ticker = yf.Ticker(symbol)
            raw = ticker.history(period=download_period, interval="1h")
            df = _standardize_columns(raw)

        else:
            logger.error(f"Unsupported interval: {interval}")
            return pd.DataFrame()

        if df.empty:
            logger.error(f"No data after processing for {symbol}")
            return pd.DataFrame()

        logger.info(
            f"Downloaded {len(df)} candles: "
            f"{df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')}"
        )

        if save:
            _save_csv(df, instrument, interval)

        return df

    except Exception as e:
        logger.error(f"Download failed for {instrument}: {e}")
        return pd.DataFrame()


def load_csv(
    filepath: str,
    date_column: str | None = None,
    date_format: str | None = None,
) -> pd.DataFrame:
    """Load OHLCV data from a user-provided CSV file.

    Automatically detects common column naming conventions.

    Args:
        filepath: Path to the CSV file.
        date_column: Name of the date/time column (auto-detected if None).
        date_format: Date format string (auto-detected if None).

    Returns:
        Standardized OHLCV DataFrame.
    """
    path = Path(filepath)
    if not path.exists():
        logger.error(f"CSV file not found: {filepath}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(filepath)
        logger.info(f"Loaded CSV: {len(df)} rows, columns: {list(df.columns)}")

        # Auto-detect and rename columns
        df = _auto_detect_columns(df, date_column, date_format)

        if df.empty:
            return df

        logger.info(
            f"CSV loaded: {len(df)} candles, "
            f"{df.index[0]} → {df.index[-1]}"
        )
        return df

    except Exception as e:
        logger.error(f"Failed to load CSV: {e}")
        return pd.DataFrame()


def load_or_download(
    instrument: str,
    interval: str = "4h",
    years: int = 3,
) -> pd.DataFrame:
    """Load cached CSV if available, otherwise download.

    Args:
        instrument: OANDA-style instrument.
        interval: Candle interval.
        years: Years of data to download if not cached.

    Returns:
        OHLCV DataFrame.
    """
    # Check for cached file
    csv_path = DATA_DIR / f"{instrument}_{interval}.csv"
    if csv_path.exists():
        df = load_csv(str(csv_path))
        if not df.empty:
            logger.info(f"Using cached data: {csv_path}")
            return df

    # Download fresh
    return download_forex_data(instrument, years=years, interval=interval)


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize Yahoo Finance columns to our format."""
    if df.empty:
        return df

    df = df.copy()

    # Yahoo Finance uses: Open, High, Low, Close, Volume
    rename_map = {}
    for col in df.columns:
        col_lower = col.lower().strip()
        if col_lower == "open":
            rename_map[col] = "open"
        elif col_lower == "high":
            rename_map[col] = "high"
        elif col_lower == "low":
            rename_map[col] = "low"
        elif col_lower == "close":
            rename_map[col] = "close"
        elif col_lower == "volume":
            rename_map[col] = "volume"

    df = df.rename(columns=rename_map)

    # Ensure required columns exist
    required = ["open", "high", "low", "close"]
    for col in required:
        if col not in df.columns:
            logger.error(f"Missing required column: {col}")
            return pd.DataFrame()

    if "volume" not in df.columns:
        df["volume"] = 0

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Remove timezone info for consistency
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # Drop rows with NaN in OHLC
    df = df.dropna(subset=required)

    # Keep only needed columns
    df = df[["open", "high", "low", "close", "volume"]]

    return df


def _resample_to_4h(df: pd.DataFrame) -> pd.DataFrame:
    """Resample 1-hour data to 4-hour candles."""
    df = _standardize_columns(df)
    if df.empty:
        return df

    resampled = df.resample("4h").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()

    return resampled


def _auto_detect_columns(
    df: pd.DataFrame,
    date_column: str | None = None,
    date_format: str | None = None,
) -> pd.DataFrame:
    """Auto-detect column names and set up proper index."""
    df = df.copy()

    # Common date column names
    date_candidates = [
        "date", "datetime", "time", "timestamp", "Date", "DateTime",
        "Time", "Timestamp", "DATE", "DATETIME",
    ]

    # Find date column
    if date_column:
        date_col = date_column
    else:
        date_col = None
        for candidate in date_candidates:
            if candidate in df.columns:
                date_col = candidate
                break
        # Try first column if no match
        if date_col is None and not isinstance(df.index, pd.DatetimeIndex):
            first_col = df.columns[0]
            try:
                pd.to_datetime(df[first_col].head())
                date_col = first_col
            except (ValueError, TypeError):
                pass

    # Set date index
    if date_col and date_col in df.columns:
        if date_format:
            df[date_col] = pd.to_datetime(df[date_col], format=date_format)
        else:
            df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)

    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except (ValueError, TypeError):
            logger.error("Could not detect date column. Specify date_column parameter.")
            return pd.DataFrame()

    return _standardize_columns(df)


def _save_csv(df: pd.DataFrame, instrument: str, interval: str) -> None:
    """Save DataFrame to CSV cache."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    filepath = DATA_DIR / f"{instrument}_{interval}.csv"
    df.to_csv(filepath)
    logger.info(f"Saved: {filepath} ({len(df)} candles)")
