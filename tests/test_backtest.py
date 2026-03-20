"""Tests for backtest engine."""

import numpy as np
import pandas as pd
import pytest

from src.models.backtest_engine import BacktestEngine, BacktestResult


def _create_trending_data(n: int = 500, trend: str = "up") -> pd.DataFrame:
    """Create trending OHLCV data."""
    np.random.seed(42)
    base = 150.0
    if trend == "up":
        drift = 0.0003
    elif trend == "down":
        drift = -0.0003
    else:
        drift = 0.0

    returns = np.random.normal(drift, 0.005, n)
    close = base * np.cumprod(1 + returns)
    high = close * (1 + np.abs(np.random.normal(0, 0.003, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.003, n)))

    df = pd.DataFrame({
        "open": close * (1 + np.random.normal(0, 0.001, n)),
        "high": high,
        "low": low,
        "close": close,
        "volume": np.random.randint(100, 1000, n),
    })
    df.index = pd.date_range("2023-01-01", periods=n, freq="4h")
    return df


class TestBacktestEngine:
    """Test suite for BacktestEngine."""

    def setup_method(self) -> None:
        self.engine = BacktestEngine({
            "initial_capital": 1_000_000,
            "commission_per_trade": 0,
            "slippage_pips": 0.5,
        })

    def test_empty_signals(self) -> None:
        """Test backtest with no signals."""
        df = _create_trending_data()
        result = self.engine.run(df, [], pip_value=0.01)
        assert result.total_trades == 0
        assert result.total_pnl == 0

    def test_single_winning_trade(self) -> None:
        """Test backtest with a single winning BUY trade."""
        df = _create_trending_data(trend="up")
        entry_price = float(df["close"].iloc[50])
        sl = entry_price * 0.99
        tp = entry_price * 1.02

        signals = [{
            "bar_index": 50,
            "direction": "BUY",
            "entry_price": entry_price,
            "stop_loss": sl,
            "take_profit": tp,
            "pattern": "test",
        }]

        result = self.engine.run(df, signals, pip_value=0.01)
        assert result.total_trades >= 1

    def test_single_losing_trade(self) -> None:
        """Test backtest with a losing trade."""
        df = _create_trending_data(trend="down")
        entry_price = float(df["close"].iloc[50])
        sl = entry_price * 0.995  # Tight stop
        tp = entry_price * 1.05   # Far TP

        signals = [{
            "bar_index": 50,
            "direction": "BUY",
            "entry_price": entry_price,
            "stop_loss": sl,
            "take_profit": tp,
            "pattern": "test",
        }]

        result = self.engine.run(df, signals, pip_value=0.01)
        assert result.total_trades >= 1

    def test_multiple_trades(self) -> None:
        """Test backtest with multiple signals."""
        df = _create_trending_data(n=500)

        signals = []
        for i in range(50, 400, 50):
            entry = float(df["close"].iloc[i])
            signals.append({
                "bar_index": i,
                "direction": "BUY",
                "entry_price": entry,
                "stop_loss": entry * 0.99,
                "take_profit": entry * 1.02,
                "pattern": "test",
            })

        result = self.engine.run(df, signals, pip_value=0.01)
        assert result.total_trades > 0
        assert result.winning_trades + result.losing_trades == result.total_trades

    def test_metrics_calculation(self) -> None:
        """Test that all metrics are calculated correctly."""
        df = _create_trending_data(n=500)

        signals = []
        for i in range(50, 400, 30):
            entry = float(df["close"].iloc[i])
            signals.append({
                "bar_index": i,
                "direction": "BUY",
                "entry_price": entry,
                "stop_loss": entry * 0.99,
                "take_profit": entry * 1.01,
                "pattern": "test",
            })

        result = self.engine.run(df, signals, pip_value=0.01)

        if result.total_trades > 0:
            assert 0 <= result.win_rate <= 1.0
            assert result.profit_factor >= 0
            assert result.max_drawdown >= 0
            assert len(result.equity_curve) > 0

    def test_backtest_result_meets_criteria(self) -> None:
        """Test criteria checking method."""
        result = BacktestResult(
            profit_factor=1.5,
            win_rate=0.40,
        )
        assert result.meets_criteria(min_pf=1.2, min_wr=0.35)

        result2 = BacktestResult(
            profit_factor=1.0,
            win_rate=0.30,
        )
        assert not result2.meets_criteria(min_pf=1.2, min_wr=0.35)

    def test_sell_trade(self) -> None:
        """Test SELL direction trade."""
        df = _create_trending_data(trend="down")
        entry = float(df["close"].iloc[50])

        signals = [{
            "bar_index": 50,
            "direction": "SELL",
            "entry_price": entry,
            "stop_loss": entry * 1.01,
            "take_profit": entry * 0.98,
            "pattern": "test",
        }]

        result = self.engine.run(df, signals, pip_value=0.01)
        assert result.total_trades >= 1
