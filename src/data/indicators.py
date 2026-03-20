"""Technical indicators calculation module.

Computes all technical indicators needed for the trading strategies,
following AI Sennin's methodology.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger
from ta.momentum import RSIIndicator, StochRSIIndicator
from ta.trend import (
    ADXIndicator,
    EMAIndicator,
    MACD,
    SMAIndicator,
)
from ta.volatility import AverageTrueRange, BollingerBands


class TechnicalAnalyzer:
    """Calculate technical indicators on OHLCV DataFrames."""

    def __init__(self, config: dict | None = None) -> None:
        """Initialize with indicator configuration.

        Args:
            config: Indicator parameters from trading_config.yaml.
        """
        self.config = config or {}

    def add_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all configured technical indicators to the DataFrame.

        Args:
            df: OHLCV DataFrame.

        Returns:
            DataFrame with added indicator columns.
        """
        if df.empty:
            return df

        df = df.copy()
        df = self.add_moving_averages(df)
        df = self.add_rsi(df)
        df = self.add_macd(df)
        df = self.add_bollinger_bands(df)
        df = self.add_atr(df)
        df = self.add_adx(df)
        df = self.add_support_resistance(df)
        df = self.add_currency_strength_proxy(df)

        return df

    def add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add SMA and EMA indicators.

        Args:
            df: OHLCV DataFrame.

        Returns:
            DataFrame with MA columns added.
        """
        sma_periods = self.config.get("sma_periods", [20, 50, 100, 200])
        ema_periods = self.config.get("ema_periods", [9, 21, 55, 200])

        for period in sma_periods:
            if len(df) >= period:
                indicator = SMAIndicator(close=df["close"], window=period)
                df[f"sma_{period}"] = indicator.sma_indicator()

        for period in ema_periods:
            if len(df) >= period:
                indicator = EMAIndicator(close=df["close"], window=period)
                df[f"ema_{period}"] = indicator.ema_indicator()

        return df

    def add_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add RSI indicator.

        Args:
            df: OHLCV DataFrame.

        Returns:
            DataFrame with RSI column.
        """
        period = self.config.get("rsi_period", 14)
        if len(df) >= period:
            rsi = RSIIndicator(close=df["close"], window=period)
            df["rsi"] = rsi.rsi()
        return df

    def add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add MACD indicator.

        Args:
            df: OHLCV DataFrame.

        Returns:
            DataFrame with MACD columns.
        """
        fast = self.config.get("macd_fast", 12)
        slow = self.config.get("macd_slow", 26)
        signal = self.config.get("macd_signal", 9)

        if len(df) >= slow:
            macd = MACD(
                close=df["close"],
                window_fast=fast,
                window_slow=slow,
                window_sign=signal,
            )
            df["macd"] = macd.macd()
            df["macd_signal"] = macd.macd_signal()
            df["macd_histogram"] = macd.macd_diff()

        return df

    def add_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Bollinger Bands.

        Args:
            df: OHLCV DataFrame.

        Returns:
            DataFrame with BB columns.
        """
        period = self.config.get("bb_period", 20)
        std = self.config.get("bb_std", 2.0)

        if len(df) >= period:
            bb = BollingerBands(close=df["close"], window=period, window_dev=std)
            df["bb_upper"] = bb.bollinger_hband()
            df["bb_middle"] = bb.bollinger_mavg()
            df["bb_lower"] = bb.bollinger_lband()
            df["bb_width"] = bb.bollinger_wband()
            df["bb_pct"] = bb.bollinger_pband()

        return df

    def add_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Average True Range for stop-loss calculation.

        Args:
            df: OHLCV DataFrame.

        Returns:
            DataFrame with ATR column.
        """
        period = self.config.get("atr_period", 14)
        if len(df) >= period:
            atr = AverageTrueRange(
                high=df["high"], low=df["low"], close=df["close"], window=period
            )
            df["atr"] = atr.average_true_range()

        return df

    def add_adx(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add ADX (Average Directional Index) for trend strength.

        Args:
            df: OHLCV DataFrame.

        Returns:
            DataFrame with ADX columns.
        """
        period = self.config.get("adx_period", 14)
        if len(df) >= period * 2:
            adx = ADXIndicator(
                high=df["high"], low=df["low"], close=df["close"], window=period
            )
            df["adx"] = adx.adx()
            df["adx_pos"] = adx.adx_pos()
            df["adx_neg"] = adx.adx_neg()

        return df

    def add_support_resistance(
        self, df: pd.DataFrame, lookback: int = 20
    ) -> pd.DataFrame:
        """Calculate support and resistance levels from recent swing highs/lows.

        Args:
            df: OHLCV DataFrame.
            lookback: Lookback period for finding swings.

        Returns:
            DataFrame with support/resistance columns.
        """
        if len(df) < lookback * 2:
            return df

        # Rolling support (recent lows)
        df["support"] = df["low"].rolling(window=lookback).min()

        # Rolling resistance (recent highs)
        df["resistance"] = df["high"].rolling(window=lookback).max()

        # Pivot points
        df["pivot"] = (df["high"] + df["low"] + df["close"]) / 3
        df["r1"] = 2 * df["pivot"] - df["low"]
        df["s1"] = 2 * df["pivot"] - df["high"]

        return df

    def add_currency_strength_proxy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add a proxy for relative currency strength using price momentum.

        Args:
            df: OHLCV DataFrame.

        Returns:
            DataFrame with strength proxy columns.
        """
        # Rate of change as a strength proxy
        for period in [5, 10, 20]:
            if len(df) >= period:
                df[f"roc_{period}"] = df["close"].pct_change(periods=period) * 100

        return df

    def detect_divergence(
        self, df: pd.DataFrame, indicator: str = "rsi", lookback: int = 20
    ) -> pd.Series:
        """Detect price-indicator divergence.

        Args:
            df: DataFrame with price and indicator data.
            indicator: Indicator column name.
            lookback: Lookback period.

        Returns:
            Series with divergence signals (-1=bearish, 0=none, 1=bullish).
        """
        if indicator not in df.columns:
            return pd.Series(0, index=df.index)

        signals = pd.Series(0, index=df.index)

        for i in range(lookback, len(df)):
            window = df.iloc[i - lookback:i + 1]

            # Bullish divergence: price lower low, indicator higher low
            price_ll = window["close"].iloc[-1] < window["close"].min()
            ind_hl = window[indicator].iloc[-1] > window[indicator].iloc[
                window["close"].idxmin() - window.index[0]
                if isinstance(window.index[0], int) else 0
            ]

            if price_ll and ind_hl:
                signals.iloc[i] = 1

            # Bearish divergence: price higher high, indicator lower high
            price_hh = window["close"].iloc[-1] > window["close"].max()
            ind_lh = window[indicator].iloc[-1] < window[indicator].iloc[
                window["close"].idxmax() - window.index[0]
                if isinstance(window.index[0], int) else 0
            ]

            if price_hh and ind_lh:
                signals.iloc[i] = -1

        return signals

    def get_trend_direction(self, df: pd.DataFrame) -> str:
        """Determine overall trend direction using multiple indicators.

        Args:
            df: DataFrame with indicators added.

        Returns:
            'BULLISH', 'BEARISH', or 'NEUTRAL'.
        """
        if df.empty or len(df) < 200:
            return "NEUTRAL"

        latest = df.iloc[-1]
        score = 0

        # MA alignment
        if "sma_50" in df.columns and "sma_200" in df.columns:
            if latest["sma_50"] > latest["sma_200"]:
                score += 1
            else:
                score -= 1

        # Price vs SMA200
        if "sma_200" in df.columns:
            if latest["close"] > latest["sma_200"]:
                score += 1
            else:
                score -= 1

        # MACD
        if "macd_histogram" in df.columns:
            if latest["macd_histogram"] > 0:
                score += 1
            else:
                score -= 1

        # ADX with direction
        if "adx" in df.columns and "adx_pos" in df.columns and "adx_neg" in df.columns:
            if latest["adx"] > 25:
                if latest["adx_pos"] > latest["adx_neg"]:
                    score += 1
                else:
                    score -= 1

        if score >= 2:
            return "BULLISH"
        elif score <= -2:
            return "BEARISH"
        return "NEUTRAL"
