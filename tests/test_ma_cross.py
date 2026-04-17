"""MovingAverageCrossStrategy のユニットテスト。"""

import numpy as np
import pandas as pd
import pytest

from src.strategies import MovingAverageCrossStrategy


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """テスト用のダミー OHLCV データ（200日分）。"""
    np.random.seed(0)
    dates = pd.date_range("2022-01-01", periods=200, freq="D")
    close = 100 + np.cumsum(np.random.randn(200))
    return pd.DataFrame(
        {
            "Open": close * 0.99,
            "High": close * 1.01,
            "Low": close * 0.98,
            "Close": close,
            "Volume": np.random.randint(1_000_000, 5_000_000, 200),
        },
        index=dates,
    )


class TestInit:
    def test_invalid_windows(self) -> None:
        with pytest.raises(ValueError):
            MovingAverageCrossStrategy(short_window=50, long_window=20)

    def test_valid_windows(self) -> None:
        strategy = MovingAverageCrossStrategy(short_window=10, long_window=30)
        assert strategy.short_window == 10
        assert strategy.long_window == 30


class TestGenerateSignals:
    def test_required_columns(self, sample_df: pd.DataFrame) -> None:
        strategy = MovingAverageCrossStrategy()
        result = strategy.generate_signals(sample_df)
        assert {"SMA_short", "SMA_long", "signal", "position"}.issubset(result.columns)

    def test_signal_values(self, sample_df: pd.DataFrame) -> None:
        strategy = MovingAverageCrossStrategy()
        result = strategy.generate_signals(sample_df)
        assert set(result["signal"].unique()).issubset({-1, 0, 1})

    def test_position_values(self, sample_df: pd.DataFrame) -> None:
        strategy = MovingAverageCrossStrategy()
        result = strategy.generate_signals(sample_df)
        assert set(result["position"].unique()).issubset({0, 1})


class TestBacktest:
    def test_cumulative_starts_near_one(self, sample_df: pd.DataFrame) -> None:
        strategy = MovingAverageCrossStrategy()
        result = strategy.backtest(sample_df)
        # pct_change() で先頭行はNaNになるため、最初の有効値を確認する
        first_valid = result["cumulative_returns"].dropna().iloc[0]
        assert pytest.approx(first_valid, abs=0.05) == 1.0

    def test_no_nan_in_cumulative(self, sample_df: pd.DataFrame) -> None:
        strategy = MovingAverageCrossStrategy()
        result = strategy.backtest(sample_df)
        # shift(1) により先頭行はNaNになるが、2行目以降はNaNなし
        assert not result["cumulative_strategy"].iloc[1:].isna().any()


class TestSummary:
    def test_summary_keys(self, sample_df: pd.DataFrame) -> None:
        strategy = MovingAverageCrossStrategy()
        summary = strategy.summary(sample_df)
        assert set(summary.keys()) == {
            "total_return",
            "buy_and_hold_return",
            "num_trades",
            "sharpe_ratio",
        }

    def test_num_trades_positive(self, sample_df: pd.DataFrame) -> None:
        strategy = MovingAverageCrossStrategy()
        summary = strategy.summary(sample_df)
        assert summary["num_trades"] >= 0
