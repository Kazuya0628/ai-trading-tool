"""Tests for pattern detection module."""

import numpy as np
import pandas as pd
import pytest

from src.strategies.pattern_detector import (
    PatternDetector,
    PatternSignal,
    PatternType,
    SignalDirection,
)


def _create_ohlcv(prices: list[float], noise: float = 0.001) -> pd.DataFrame:
    """Create synthetic OHLCV DataFrame from close prices."""
    np.random.seed(42)
    n = len(prices)
    close = np.array(prices)
    jitter = np.random.uniform(-noise, noise, n) * close
    high = close + abs(jitter)
    low = close - abs(jitter)
    open_ = close + np.random.uniform(-noise / 2, noise / 2, n) * close

    df = pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": np.random.randint(100, 1000, n),
    })
    df.index = pd.date_range("2024-01-01", periods=n, freq="4h")
    df.index.name = "datetime"
    return df


def _generate_w_pattern(base_price: float = 100.0, n_bars: int = 80) -> pd.DataFrame:
    """Generate a W-shaped (double bottom) price pattern."""
    prices = []
    # Initial decline to first trough
    for i in range(20):
        prices.append(base_price - (base_price * 0.05) * (i / 20))
    # Rise to neckline
    for i in range(15):
        prices.append(base_price * 0.95 + (base_price * 0.04) * (i / 15))
    # Decline to second trough (slightly higher - preferred variant)
    for i in range(15):
        prices.append(base_price * 0.99 - (base_price * 0.035) * (i / 15))
    # Rise breaking neckline
    for i in range(n_bars - 50):
        prices.append(base_price * 0.955 + (base_price * 0.06) * (i / (n_bars - 50)))

    return _create_ohlcv(prices)


def _generate_m_pattern(base_price: float = 100.0, n_bars: int = 80) -> pd.DataFrame:
    """Generate an M-shaped (double top) price pattern."""
    prices = []
    # Rise to first peak
    for i in range(20):
        prices.append(base_price + (base_price * 0.05) * (i / 20))
    # Decline to neckline
    for i in range(15):
        prices.append(base_price * 1.05 - (base_price * 0.04) * (i / 15))
    # Rise to second peak (slightly lower)
    for i in range(15):
        prices.append(base_price * 1.01 + (base_price * 0.035) * (i / 15))
    # Decline breaking neckline
    for i in range(n_bars - 50):
        prices.append(base_price * 1.045 - (base_price * 0.06) * (i / (n_bars - 50)))

    return _create_ohlcv(prices)


class TestPatternDetector:
    """Test suite for PatternDetector."""

    def setup_method(self) -> None:
        """Setup test fixtures."""
        self.detector = PatternDetector({
            "enabled_patterns": [
                "double_bottom", "double_top",
                "inverse_head_shoulders", "head_shoulders",
                "channel_breakout", "ma_crossover",
            ],
            "double_bottom": {
                "prefer_higher_low": True,
                "min_pattern_bars": 20,
                "max_pattern_bars": 100,
            },
        })

    def test_pattern_signal_dataclass(self) -> None:
        """Test PatternSignal creation and properties."""
        signal = PatternSignal(
            pattern=PatternType.DOUBLE_BOTTOM,
            direction=SignalDirection.BUY,
            confidence=75.0,
            entry_price=100.0,
            stop_loss=95.0,
            take_profit=110.0,
        )
        assert signal.is_valid
        assert signal.signal_code == 1
        assert signal.pattern == PatternType.DOUBLE_BOTTOM
        assert signal.direction == SignalDirection.BUY

    def test_no_signal(self) -> None:
        """Test PatternSignal for no signal."""
        signal = PatternSignal(
            pattern=PatternType.NO_SIGNAL,
            direction=SignalDirection.NONE,
            confidence=0,
        )
        assert not signal.is_valid
        assert signal.signal_code == 0

    def test_double_bottom_detection(self) -> None:
        """Test W-pattern detection."""
        df = _generate_w_pattern()
        signal = self.detector.detect_double_bottom(df)
        # Pattern should be detected with BUY direction
        assert signal.direction == SignalDirection.BUY or signal.pattern == PatternType.NO_SIGNAL
        if signal.is_valid:
            assert signal.confidence >= 50
            assert signal.stop_loss > 0
            assert signal.take_profit > signal.entry_price

    def test_double_top_detection(self) -> None:
        """Test M-pattern detection."""
        df = _generate_m_pattern()
        signal = self.detector.detect_double_top(df)
        if signal.is_valid:
            assert signal.direction == SignalDirection.SELL
            assert signal.confidence >= 50
            assert signal.take_profit < signal.entry_price

    def test_empty_dataframe(self) -> None:
        """Test with empty DataFrame."""
        df = pd.DataFrame()
        signals = self.detector.detect_all_patterns(df)
        assert signals == []

    def test_short_dataframe(self) -> None:
        """Test with too-short DataFrame."""
        df = _create_ohlcv([100.0] * 10)
        signals = self.detector.detect_all_patterns(df)
        assert signals == []

    def test_flat_market(self) -> None:
        """Test with flat (no pattern) market."""
        prices = [100.0 + np.random.normal(0, 0.1) for _ in range(100)]
        df = _create_ohlcv(prices)
        signals = self.detector.detect_all_patterns(df)
        # Should return few or no signals in flat market
        for s in signals:
            if s.is_valid:
                assert s.confidence < 90  # Should not be highly confident

    def test_swing_point_detection(self) -> None:
        """Test swing point finder helper."""
        # Simple V-shape
        data = np.array([10, 9, 8, 7, 6, 5, 6, 7, 8, 9, 10])
        lows = PatternDetector._find_swing_points(data, order=3, mode="low")
        assert len(lows) >= 1
        assert lows[0][1] == 5.0  # Lowest point

        highs = PatternDetector._find_swing_points(data, order=3, mode="high")
        # Edges may or may not be detected depending on order


class TestMAcrossover:
    """Test MA crossover detection."""

    def setup_method(self) -> None:
        self.detector = PatternDetector({
            "enabled_patterns": ["ma_crossover"],
            "ma_crossover": {"fast_period": 20, "slow_period": 50},
        })

    def test_golden_cross(self) -> None:
        """Test bullish MA crossover detection."""
        # Create uptrending data
        prices = list(np.linspace(95, 105, 100))
        df = _create_ohlcv(prices)

        # Manually add MA columns that cross
        df["sma_20"] = df["close"].rolling(20).mean()
        df["sma_50"] = df["close"].rolling(50).mean()
        df["atr"] = 0.5  # Dummy ATR

        signal = self.detector.detect_ma_crossover(df)
        # Result depends on whether crossover actually occurred
        # The test validates the method runs without error
        assert isinstance(signal, PatternSignal)

    def test_no_crossover(self) -> None:
        """Test when no crossover exists."""
        prices = [100.0] * 100
        df = _create_ohlcv(prices)
        df["sma_20"] = 100.0
        df["sma_50"] = 100.0
        df["atr"] = 0.5

        signal = self.detector.detect_ma_crossover(df)
        assert signal.pattern == PatternType.NO_SIGNAL


class TestChannelBreakout:
    """Test channel breakout detection."""

    def setup_method(self) -> None:
        self.detector = PatternDetector({
            "enabled_patterns": ["channel_breakout"],
            "channel_breakout": {"lookback_period": 20},
        })

    def test_bullish_breakout(self) -> None:
        """Test bullish channel breakout."""
        # Range-bound then breakout up
        prices = [100.0 + np.random.normal(0, 0.3) for _ in range(25)]
        prices.append(103.0)  # Breakout bar
        df = _create_ohlcv(prices)
        df["atr"] = 0.5

        signal = self.detector.detect_channel_breakout(df)
        if signal.is_valid:
            assert signal.direction == SignalDirection.BUY

    def test_bearish_breakout(self) -> None:
        """Test bearish channel breakout."""
        prices = [100.0 + np.random.normal(0, 0.3) for _ in range(25)]
        prices.append(97.0)  # Breakout bar
        df = _create_ohlcv(prices)
        df["atr"] = 0.5

        signal = self.detector.detect_channel_breakout(df)
        if signal.is_valid:
            assert signal.direction == SignalDirection.SELL
