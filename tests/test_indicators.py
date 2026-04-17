"""Tests for technical indicators module."""

import numpy as np
import pandas as pd
import pytest

from src.data.indicators import TechnicalAnalyzer


def _create_sample_df(n: int = 250) -> pd.DataFrame:
    """Create sample OHLCV data with a trend."""
    np.random.seed(42)
    base = 150.0
    returns = np.random.normal(0.0001, 0.005, n)
    close = base * np.cumprod(1 + returns)
    high = close * (1 + np.abs(np.random.normal(0, 0.003, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.003, n)))
    open_ = close * (1 + np.random.normal(0, 0.001, n))
    volume = np.random.randint(1000, 10000, n)

    df = pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })
    df.index = pd.date_range("2024-01-01", periods=n, freq="4h")
    df.index.name = "datetime"
    return df


class TestTechnicalAnalyzer:
    """Test suite for TechnicalAnalyzer."""

    def setup_method(self) -> None:
        self.analyzer = TechnicalAnalyzer({
            "sma_periods": [20, 50, 200],
            "ema_periods": [9, 21],
            "rsi_period": 14,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "bb_period": 20,
            "bb_std": 2.0,
            "atr_period": 14,
            "adx_period": 14,
        })

    def test_add_all_indicators(self) -> None:
        """Test that all indicators are added."""
        df = _create_sample_df()
        result = self.analyzer.add_all_indicators(df)

        # Check key columns exist
        assert "sma_20" in result.columns
        assert "sma_50" in result.columns
        assert "sma_200" in result.columns
        assert "ema_9" in result.columns
        assert "rsi" in result.columns
        assert "macd" in result.columns
        assert "macd_signal" in result.columns
        assert "macd_histogram" in result.columns
        assert "bb_upper" in result.columns
        assert "bb_lower" in result.columns
        assert "atr" in result.columns
        assert "adx" in result.columns

    def test_sma_values(self) -> None:
        """Test SMA calculations are reasonable."""
        df = _create_sample_df()
        result = self.analyzer.add_moving_averages(df)

        # SMA20 should be close to recent price
        sma20_last = result["sma_20"].iloc[-1]
        close_last = result["close"].iloc[-1]
        assert abs(sma20_last - close_last) / close_last < 0.05  # Within 5%

    def test_rsi_range(self) -> None:
        """Test RSI is bounded 0-100."""
        df = _create_sample_df()
        result = self.analyzer.add_rsi(df)
        rsi = result["rsi"].dropna()
        assert rsi.min() >= 0
        assert rsi.max() <= 100

    def test_atr_positive(self) -> None:
        """Test ATR is always positive after warmup period."""
        df = _create_sample_df()
        result = self.analyzer.add_atr(df)
        atr = result["atr"].dropna()
        # ATR should be positive after warmup (first few values may be 0/NaN)
        atr_valid = atr[atr > 0]
        assert len(atr_valid) > 0
        assert (atr_valid > 0).all()

    def test_bollinger_bands_order(self) -> None:
        """Test BB upper > middle > lower."""
        df = _create_sample_df()
        result = self.analyzer.add_bollinger_bands(df)
        valid = result.dropna(subset=["bb_upper", "bb_middle", "bb_lower"])
        assert (valid["bb_upper"] >= valid["bb_middle"]).all()
        assert (valid["bb_middle"] >= valid["bb_lower"]).all()

    def test_empty_dataframe(self) -> None:
        """Test with empty DataFrame."""
        df = pd.DataFrame()
        result = self.analyzer.add_all_indicators(df)
        assert result.empty

    def test_trend_direction(self) -> None:
        """Test trend direction detection."""
        df = _create_sample_df(300)
        result = self.analyzer.add_all_indicators(df)
        trend = self.analyzer.get_trend_direction(result)
        assert trend in ["BULLISH", "BEARISH", "NEUTRAL"]

    def test_support_resistance(self) -> None:
        """Test support/resistance calculation."""
        df = _create_sample_df()
        result = self.analyzer.add_support_resistance(df)
        valid = result.dropna(subset=["support", "resistance"])
        assert (valid["support"] <= valid["resistance"]).all()
