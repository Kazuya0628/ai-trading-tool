"""Backtesting Engine.

AI Sennin's approach:
- Test thousands of hypotheses quickly
- Minimum 3 years of historical data
- Focus on Profit Factor >= 1.2 and consistent performance
- Walk-forward validation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class BacktestTrade:
    """Represents a single backtest trade."""
    entry_bar: int
    exit_bar: int
    direction: str
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    size: float
    pnl: float = 0.0
    pnl_pips: float = 0.0
    pattern: str = ""
    exit_reason: str = ""


@dataclass
class BacktestResult:
    """Backtest performance metrics."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    calmar_ratio: float = 0.0
    avg_rr_ratio: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    trades: list[BacktestTrade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)

    def meets_criteria(self, min_pf: float = 1.2, min_wr: float = 0.35) -> bool:
        """Check if backtest meets AI Sennin's minimum criteria."""
        return self.profit_factor >= min_pf and self.win_rate >= min_wr

    def summary(self) -> dict[str, Any]:
        return {
            "total_trades": self.total_trades,
            "win_rate": f"{self.win_rate:.1%}",
            "profit_factor": f"{self.profit_factor:.2f}",
            "total_pnl": f"{self.total_pnl:.2f}",
            "avg_win": f"{self.avg_win:.2f}",
            "avg_loss": f"{self.avg_loss:.2f}",
            "max_drawdown": f"{self.max_drawdown:.2f}",
            "max_drawdown_pct": f"{self.max_drawdown_pct:.1%}",
            "sharpe_ratio": f"{self.sharpe_ratio:.2f}",
            "max_consecutive_losses": self.max_consecutive_losses,
        }


class BacktestEngine:
    """Run backtests on historical data with pattern-based strategies."""

    def __init__(self, config: dict | None = None) -> None:
        """Initialize BacktestEngine.

        Args:
            config: Backtesting configuration.
        """
        self.config = config or {}
        self.initial_capital = self.config.get("initial_capital", 1_000_000)
        self.commission = self.config.get("commission_per_trade", 0)
        self.slippage_pips = self.config.get("slippage_pips", 0.5)

    def run(
        self,
        df: pd.DataFrame,
        signals: list[dict[str, Any]],
        pip_value: float = 0.01,
    ) -> BacktestResult:
        """Run backtest on historical data with pre-computed signals.

        Args:
            df: OHLCV DataFrame.
            signals: List of signal dicts with keys:
                bar_index, direction, entry_price, stop_loss, take_profit, pattern
            pip_value: Pip value for the currency pair.

        Returns:
            BacktestResult with performance metrics.
        """
        trades: list[BacktestTrade] = []
        equity = self.initial_capital
        equity_curve = [equity]
        peak_equity = equity

        close = df["close"].values
        high = df["high"].values
        low = df["low"].values

        position = None  # Active position

        for signal in sorted(signals, key=lambda s: s["bar_index"]):
            bar_idx = signal["bar_index"]

            if bar_idx >= len(df) - 1:
                continue

            # Skip if already in a position
            if position is not None:
                continue

            direction = signal["direction"]
            entry_price = signal["entry_price"]
            sl = signal["stop_loss"]
            tp = signal["take_profit"]
            pattern = signal.get("pattern", "")

            # Apply slippage
            slippage = self.slippage_pips * pip_value
            if direction == "BUY":
                entry_price += slippage
            else:
                entry_price -= slippage

            # Position sizing (1% risk)
            risk_per_trade = equity * 0.01
            stop_distance = abs(entry_price - sl)
            if stop_distance == 0:
                continue
            size = risk_per_trade / stop_distance

            position = {
                "entry_bar": bar_idx,
                "direction": direction,
                "entry_price": entry_price,
                "stop_loss": sl,
                "take_profit": tp,
                "size": size,
                "pattern": pattern,
            }

            # Simulate forward from entry
            for i in range(bar_idx + 1, len(df)):
                exit_reason = ""
                exit_price = 0

                if direction == "BUY":
                    # Check stop-loss
                    if low[i] <= sl:
                        exit_price = sl
                        exit_reason = "stop_loss"
                    # Check take-profit
                    elif high[i] >= tp:
                        exit_price = tp
                        exit_reason = "take_profit"
                elif direction == "SELL":
                    if high[i] >= sl:
                        exit_price = sl
                        exit_reason = "stop_loss"
                    elif low[i] <= tp:
                        exit_price = tp
                        exit_reason = "take_profit"

                if exit_reason:
                    # Calculate P&L
                    if direction == "BUY":
                        pnl = (exit_price - entry_price) * size
                    else:
                        pnl = (entry_price - exit_price) * size

                    pnl -= self.commission
                    pnl_pips = abs(exit_price - entry_price) / pip_value

                    trade = BacktestTrade(
                        entry_bar=bar_idx,
                        exit_bar=i,
                        direction=direction,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        stop_loss=sl,
                        take_profit=tp,
                        size=size,
                        pnl=pnl,
                        pnl_pips=pnl_pips if pnl > 0 else -pnl_pips,
                        pattern=pattern,
                        exit_reason=exit_reason,
                    )
                    trades.append(trade)

                    equity += pnl
                    equity_curve.append(equity)
                    peak_equity = max(peak_equity, equity)

                    position = None
                    break

            # If no exit found, close at end
            if position is not None:
                exit_price = float(close[-1])
                if direction == "BUY":
                    pnl = (exit_price - entry_price) * size
                else:
                    pnl = (entry_price - exit_price) * size

                trade = BacktestTrade(
                    entry_bar=bar_idx,
                    exit_bar=len(df) - 1,
                    direction=direction,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    stop_loss=sl,
                    take_profit=tp,
                    size=size,
                    pnl=pnl,
                    pattern=pattern,
                    exit_reason="end_of_data",
                )
                trades.append(trade)
                equity += pnl
                equity_curve.append(equity)
                position = None

        return self._calculate_metrics(trades, equity_curve)

    def _calculate_metrics(
        self, trades: list[BacktestTrade], equity_curve: list[float]
    ) -> BacktestResult:
        """Calculate performance metrics from trades.

        Args:
            trades: List of completed trades.
            equity_curve: Equity values over time.

        Returns:
            BacktestResult with all metrics.
        """
        result = BacktestResult(trades=trades, equity_curve=equity_curve)

        if not trades:
            return result

        result.total_trades = len(trades)
        pnls = [t.pnl for t in trades]

        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        result.winning_trades = len(wins)
        result.losing_trades = len(losses)
        result.win_rate = len(wins) / len(trades) if trades else 0
        result.total_pnl = sum(pnls)
        result.avg_win = np.mean(wins) if wins else 0
        result.avg_loss = np.mean(losses) if losses else 0

        # Profit Factor
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Max Drawdown
        peak = equity_curve[0]
        max_dd = 0
        max_dd_pct = 0
        for eq in equity_curve:
            peak = max(peak, eq)
            dd = peak - eq
            dd_pct = dd / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
            max_dd_pct = max(max_dd_pct, dd_pct)

        result.max_drawdown = max_dd
        result.max_drawdown_pct = max_dd_pct

        # Sharpe Ratio (annualized)
        if len(pnls) > 1:
            returns = np.array(pnls) / self.initial_capital
            result.sharpe_ratio = (
                np.mean(returns) / np.std(returns) * np.sqrt(252)
                if np.std(returns) > 0 else 0
            )

        # Calmar Ratio
        annual_return = result.total_pnl / self.initial_capital
        result.calmar_ratio = (
            annual_return / result.max_drawdown_pct
            if result.max_drawdown_pct > 0 else 0
        )

        # Average R:R
        if result.avg_loss != 0:
            result.avg_rr_ratio = abs(result.avg_win / result.avg_loss)

        # Consecutive wins/losses
        result.max_consecutive_wins = self._max_consecutive(pnls, positive=True)
        result.max_consecutive_losses = self._max_consecutive(pnls, positive=False)

        return result

    @staticmethod
    def _max_consecutive(pnls: list[float], positive: bool) -> int:
        """Count max consecutive wins or losses."""
        max_count = 0
        current = 0
        for pnl in pnls:
            if (positive and pnl > 0) or (not positive and pnl <= 0):
                current += 1
                max_count = max(max_count, current)
            else:
                current = 0
        return max_count
