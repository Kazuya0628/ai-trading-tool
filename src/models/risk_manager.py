"""Risk Management Module.

Implements strict risk controls following AI Sennin's principles:
- Fixed risk per trade (1% of account)
- ATR-based stop-loss placement
- Maximum drawdown protection
- Daily loss limits
- Position sizing based on risk
- Adaptive risk scaling based on performance and market regime
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from src.strategies.pattern_detector import SignalDirection
from src.strategies.strategy_engine import TradeSignal


@dataclass
class PositionSize:
    """Calculated position size with risk parameters."""
    lots: float
    risk_amount: float
    stop_distance_pips: float
    pip_value: float
    account_balance: float
    risk_pct: float


@dataclass
class RiskState:
    """Current risk state tracking."""
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    peak_balance: float = 0.0
    current_drawdown_pct: float = 0.0
    daily_trade_count: int = 0
    open_position_count: int = 0
    last_reset_date: str = ""
    last_weekly_reset: str = ""
    is_trading_allowed: bool = True
    halt_reason: str = ""


@dataclass
class MarketRegime:
    """AI-determined market regime for adaptive risk adjustment."""
    regime: str = "normal"  # normal, trending, volatile, ranging, uncertain
    volatility_level: str = "medium"  # low, medium, high, extreme
    trend_strength: str = "moderate"  # weak, moderate, strong
    risk_multiplier: float = 1.0  # 0.0〜1.0, applied to position sizing
    confidence: float = 50.0
    reasoning: str = ""


class PerformanceTracker:
    """Track recent trade performance for adaptive risk decisions.

    Maintains a rolling window of trade results to detect
    losing streaks, win rate trends, and performance degradation.
    """

    def __init__(self, window_size: int = 20) -> None:
        """Initialize performance tracker.

        Args:
            window_size: Number of recent trades to track.
        """
        self.window_size = window_size
        self._results: deque[dict[str, Any]] = deque(maxlen=window_size)
        self.consecutive_losses: int = 0
        self.consecutive_wins: int = 0
        self.total_trades: int = 0

    def record(self, pnl: float, pattern: str = "", instrument: str = "") -> None:
        """Record a trade result.

        Args:
            pnl: Profit/loss amount.
            pattern: Pattern type that generated the signal.
            instrument: Currency pair.
        """
        is_win = pnl > 0
        self._results.append({
            "pnl": pnl,
            "is_win": is_win,
            "pattern": pattern,
            "instrument": instrument,
            "timestamp": datetime.now().isoformat(),
        })
        self.total_trades += 1

        if is_win:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0

    @property
    def recent_win_rate(self) -> float:
        """Calculate win rate over the rolling window."""
        if not self._results:
            return 0.5  # Default assumption
        wins = sum(1 for r in self._results if r["is_win"])
        return wins / len(self._results)

    @property
    def recent_pnl(self) -> float:
        """Total P&L over the rolling window."""
        return sum(r["pnl"] for r in self._results)

    @property
    def recent_avg_pnl(self) -> float:
        """Average P&L per trade over the rolling window."""
        if not self._results:
            return 0.0
        return self.recent_pnl / len(self._results)

    def pattern_win_rate(self, pattern: str) -> float:
        """Win rate for a specific pattern type.

        Args:
            pattern: Pattern type name.

        Returns:
            Win rate (0.0-1.0), or 0.5 if insufficient data.
        """
        pattern_trades = [r for r in self._results if r["pattern"] == pattern]
        if len(pattern_trades) < 3:
            return 0.5  # Insufficient data
        wins = sum(1 for r in pattern_trades if r["is_win"])
        return wins / len(pattern_trades)


class AdaptiveRiskController:
    """AI-driven adaptive risk adjustment.

    Dynamically scales risk based on:
    1. Recent performance (consecutive losses → reduce size)
    2. Drawdown progression (deeper DD → smaller positions)
    3. Market regime (AI-detected volatility/trend state)
    4. Pattern-specific performance (disable underperforming patterns)
    """

    def __init__(self, config: dict | None = None) -> None:
        """Initialize adaptive risk controller.

        Args:
            config: Adaptive risk configuration.
        """
        self.config = config or {}
        self.tracker = PerformanceTracker(
            window_size=self.config.get("performance_window", 20)
        )
        self.market_regime = MarketRegime()

        # Consecutive loss scaling: risk reduces by this factor per loss
        self._loss_scale_factor = self.config.get("loss_scale_factor", 0.75)
        # Maximum reduction from consecutive losses (floor)
        self._min_loss_scale = self.config.get("min_loss_scale", 0.25)
        # Max consecutive losses before forced cooldown
        self._max_consecutive_losses = self.config.get("max_consecutive_losses", 5)

        # Drawdown-based scaling thresholds
        self._dd_thresholds = self.config.get("drawdown_thresholds", [
            {"dd_pct": 3.0, "scale": 0.75},
            {"dd_pct": 5.0, "scale": 0.50},
            {"dd_pct": 7.0, "scale": 0.25},
        ])

        # Pattern disable threshold
        self._pattern_disable_wr = self.config.get("pattern_disable_win_rate", 0.25)

    def calculate_risk_multiplier(self, current_drawdown_pct: float) -> float:
        """Calculate overall risk multiplier based on all adaptive factors.

        Args:
            current_drawdown_pct: Current drawdown percentage.

        Returns:
            Risk multiplier (0.0-1.0) to apply to base position size.
        """
        # Factor 1: Consecutive loss scaling
        loss_scale = self._consecutive_loss_scale()

        # Factor 2: Drawdown-based scaling
        dd_scale = self._drawdown_scale(current_drawdown_pct)

        # Factor 3: Market regime (from AI)
        regime_scale = self.market_regime.risk_multiplier

        # Factor 4: Recent win rate momentum
        wr_scale = self._win_rate_scale()

        # Combined multiplier (multiplicative — all factors stack)
        combined = loss_scale * dd_scale * regime_scale * wr_scale
        combined = max(combined, 0.1)  # Never go below 10% of base risk
        combined = min(combined, 1.0)  # Never exceed base risk

        logger.info(
            f"Adaptive risk: loss={loss_scale:.2f} × dd={dd_scale:.2f} × "
            f"regime={regime_scale:.2f} × wr={wr_scale:.2f} = {combined:.2f}"
        )

        return round(combined, 2)

    def should_skip_pattern(self, pattern: str) -> tuple[bool, str]:
        """Check if a pattern should be skipped due to poor performance.

        Args:
            pattern: Pattern type name.

        Returns:
            Tuple of (should_skip, reason).
        """
        wr = self.tracker.pattern_win_rate(pattern)
        if wr < self._pattern_disable_wr:
            return True, (
                f"Pattern '{pattern}' disabled: win rate {wr:.0%} "
                f"< {self._pattern_disable_wr:.0%} threshold"
            )
        return False, ""

    def should_cooldown(self) -> tuple[bool, str]:
        """Check if a forced cooldown is needed due to consecutive losses.

        Returns:
            Tuple of (should_cooldown, reason).
        """
        if self.tracker.consecutive_losses >= self._max_consecutive_losses:
            return True, (
                f"Forced cooldown: {self.tracker.consecutive_losses} "
                f"consecutive losses (max={self._max_consecutive_losses})"
            )
        return False, ""

    def _consecutive_loss_scale(self) -> float:
        """Scale based on consecutive losses."""
        losses = self.tracker.consecutive_losses
        if losses == 0:
            return 1.0
        scale = self._loss_scale_factor ** losses
        return max(scale, self._min_loss_scale)

    def _drawdown_scale(self, dd_pct: float) -> float:
        """Scale based on drawdown depth."""
        scale = 1.0
        for threshold in self._dd_thresholds:
            if dd_pct >= threshold["dd_pct"]:
                scale = threshold["scale"]
        return scale

    def _win_rate_scale(self) -> float:
        """Scale based on recent win rate trend."""
        if self.tracker.total_trades < 5:
            return 1.0  # Insufficient data, use full risk
        wr = self.tracker.recent_win_rate
        if wr >= 0.55:
            return 1.0  # Performing well, full size
        elif wr >= 0.40:
            return 0.85  # Slightly below par
        elif wr >= 0.30:
            return 0.65  # Underperforming
        else:
            return 0.50  # Poor performance, halve risk

    def get_summary(self) -> dict[str, Any]:
        """Get adaptive risk state summary.

        Returns:
            Dictionary of adaptive risk metrics.
        """
        return {
            "total_trades": self.tracker.total_trades,
            "consecutive_losses": self.tracker.consecutive_losses,
            "consecutive_wins": self.tracker.consecutive_wins,
            "recent_win_rate": f"{self.tracker.recent_win_rate:.0%}",
            "recent_avg_pnl": f"{self.tracker.recent_avg_pnl:+,.0f}",
            "market_regime": self.market_regime.regime,
            "volatility": self.market_regime.volatility_level,
            "regime_risk_multiplier": self.market_regime.risk_multiplier,
        }


class RiskManager:
    """Manage trading risk and position sizing.

    Core principles (AI Sennin):
    - Risk only 1% per trade
    - A 35% win rate is fine with good R:R (e.g., 2:1)
    - Incremental exposure, never all-in
    - Math over luck: systematic risk control
    - Adaptive sizing: reduce risk in adverse conditions
    """

    def __init__(self, config: dict | None = None) -> None:
        """Initialize RiskManager.

        Args:
            config: Risk management configuration.
        """
        self.config = config or {}
        self.state = RiskState()

        # Configuration
        self.risk_per_trade_pct = self.config.get("risk_per_trade_pct", 1.0)
        self.max_position_lots = self.config.get("max_position_size_lots", 10.0)
        self.max_open_positions = self.config.get("max_open_positions", 3)
        self.max_per_pair = self.config.get("max_positions_per_pair", 1)
        self.sl_atr_multiple = self.config.get("stop_loss_atr_multiple", 2.0)
        self.min_sl_pips = self.config.get("min_stop_loss_pips", 10)
        self.max_sl_pips = self.config.get("max_stop_loss_pips", 100)
        self.tp_rr_ratio = self.config.get("take_profit_rr_ratio", 2.0)
        self.trailing_stop = self.config.get("trailing_stop_enabled", True)
        self.trailing_atr_mult = self.config.get("trailing_stop_atr_multiple", 1.5)
        self.max_daily_loss_pct = self.config.get("max_daily_loss_pct", 2.0)
        self.max_daily_trades = self.config.get("max_daily_trades", 10)
        self.max_weekly_loss_pct = self.config.get("max_weekly_loss_pct", 5.0)
        self.max_drawdown_pct = self.config.get("max_drawdown_pct", 10.0)
        self.drawdown_cooldown_hrs = self.config.get("drawdown_cooldown_hours", 24)

        # Adaptive risk controller
        self.adaptive = AdaptiveRiskController(
            self.config.get("adaptive", {})
        )

    # --------------------------------------------------
    # Pre-Trade Checks
    # --------------------------------------------------

    def can_trade(self, account_balance: float) -> tuple[bool, str]:
        """Check if trading is currently allowed.

        Args:
            account_balance: Current account balance.

        Returns:
            Tuple of (allowed, reason).
        """
        self._check_daily_reset()
        self._check_weekly_reset()

        # Update drawdown
        if account_balance > self.state.peak_balance:
            self.state.peak_balance = account_balance
        if self.state.peak_balance > 0:
            self.state.current_drawdown_pct = (
                (self.state.peak_balance - account_balance) / self.state.peak_balance * 100
            )

        # Check drawdown limit
        if self.state.current_drawdown_pct >= self.max_drawdown_pct:
            reason = (
                f"Max drawdown reached: {self.state.current_drawdown_pct:.1f}% "
                f">= {self.max_drawdown_pct}%"
            )
            self.state.is_trading_allowed = False
            self.state.halt_reason = reason
            logger.warning(f"TRADING HALTED: {reason}")
            return False, reason

        # Check daily loss
        if account_balance > 0:
            daily_loss_pct = abs(self.state.daily_pnl) / account_balance * 100
            if self.state.daily_pnl < 0 and daily_loss_pct >= self.max_daily_loss_pct:
                reason = f"Daily loss limit: {daily_loss_pct:.1f}% >= {self.max_daily_loss_pct}%"
                logger.warning(f"Trading paused: {reason}")
                return False, reason

        # Check weekly loss
        if account_balance > 0:
            weekly_loss_pct = abs(self.state.weekly_pnl) / account_balance * 100
            if self.state.weekly_pnl < 0 and weekly_loss_pct >= self.max_weekly_loss_pct:
                reason = f"Weekly loss limit: {weekly_loss_pct:.1f}% >= {self.max_weekly_loss_pct}%"
                logger.warning(f"Trading paused: {reason}")
                return False, reason

        # Check daily trade count
        if self.state.daily_trade_count >= self.max_daily_trades:
            reason = f"Daily trade limit reached: {self.state.daily_trade_count}"
            return False, reason

        # Check max open positions
        if self.state.open_position_count >= self.max_open_positions:
            reason = f"Max open positions: {self.state.open_position_count}"
            return False, reason

        self.state.is_trading_allowed = True
        self.state.halt_reason = ""
        return True, "OK"

    def validate_signal(self, signal: TradeSignal) -> tuple[bool, str]:
        """Validate a trade signal against risk rules.

        Args:
            signal: Trade signal to validate.

        Returns:
            Tuple of (valid, reason).
        """
        # Minimum confidence check
        if signal.confidence < 60:
            return False, f"Low confidence: {signal.confidence:.0f}% < 60%"

        # Minimum R:R ratio
        min_rr = self.config.get("min_risk_reward_ratio", 1.0)
        if signal.risk_reward_ratio < min_rr:
            return False, f"Poor R:R: {signal.risk_reward_ratio:.1f} < {min_rr}"

        # Stop-loss validation
        if signal.stop_loss <= 0:
            return False, "No stop-loss defined"

        # Take-profit validation
        if signal.take_profit <= 0:
            return False, "No take-profit defined"

        # Direction consistency
        if signal.direction == SignalDirection.BUY:
            if signal.stop_loss >= signal.entry_price:
                return False, "BUY: stop-loss must be below entry"
            if signal.take_profit <= signal.entry_price:
                return False, "BUY: take-profit must be above entry"
        elif signal.direction == SignalDirection.SELL:
            if signal.stop_loss <= signal.entry_price:
                return False, "SELL: stop-loss must be above entry"
            if signal.take_profit >= signal.entry_price:
                return False, "SELL: take-profit must be below entry"

        return True, "OK"

    # --------------------------------------------------
    # Position Sizing
    # --------------------------------------------------

    def calculate_position_size(
        self,
        signal: TradeSignal,
        account_balance: float,
        pip_value: float = 0.01,
    ) -> PositionSize:
        """Calculate position size based on risk parameters.

        Risk per trade = account_balance * risk_pct / 100
        Position size = risk_amount / (stop_distance * pip_value)

        Args:
            signal: Trade signal with SL/TP levels.
            account_balance: Current account balance.
            pip_value: Value per pip for the pair.

        Returns:
            PositionSize with calculated lots.
        """
        # Apply adaptive risk multiplier
        adaptive_mult = self.adaptive.calculate_risk_multiplier(
            self.state.current_drawdown_pct
        )
        adjusted_risk_pct = self.risk_per_trade_pct * adaptive_mult

        # Calculate risk amount
        risk_amount = account_balance * adjusted_risk_pct / 100

        # Calculate stop distance in pips
        stop_distance = abs(signal.entry_price - signal.stop_loss)
        stop_distance_pips = stop_distance / pip_value

        # Enforce min/max stop distance
        stop_distance_pips = max(stop_distance_pips, self.min_sl_pips)
        stop_distance_pips = min(stop_distance_pips, self.max_sl_pips)

        # Calculate lots
        if stop_distance_pips > 0 and pip_value > 0:
            lots = risk_amount / (stop_distance_pips * pip_value * 10000)
        else:
            lots = 0

        # Enforce maximum
        lots = min(lots, self.max_position_lots)
        lots = max(lots, 0.01)  # Minimum lot size

        # Round to 2 decimal places
        lots = round(lots, 2)

        return PositionSize(
            lots=lots,
            risk_amount=risk_amount,
            stop_distance_pips=stop_distance_pips,
            pip_value=pip_value,
            account_balance=account_balance,
            risk_pct=adjusted_risk_pct,
        )

    def calculate_stop_loss(
        self,
        direction: str,
        entry_price: float,
        atr: float,
        pattern_sl: float = 0,
    ) -> float:
        """Calculate stop-loss level using ATR.

        Args:
            direction: 'BUY' or 'SELL'.
            entry_price: Entry price level.
            atr: Current ATR value.
            pattern_sl: Pattern-based stop-loss (used as reference).

        Returns:
            Stop-loss price level.
        """
        atr_stop = atr * self.sl_atr_multiple

        if direction == "BUY":
            sl = entry_price - atr_stop
            # Use pattern SL if tighter and reasonable
            if pattern_sl > 0 and pattern_sl < sl:
                sl = pattern_sl
        else:
            sl = entry_price + atr_stop
            if pattern_sl > 0 and pattern_sl > sl:
                sl = pattern_sl

        return sl

    def calculate_take_profit(
        self,
        direction: str,
        entry_price: float,
        stop_loss: float,
    ) -> float:
        """Calculate take-profit level based on R:R ratio.

        Args:
            direction: 'BUY' or 'SELL'.
            entry_price: Entry price.
            stop_loss: Stop-loss level.

        Returns:
            Take-profit price level.
        """
        risk = abs(entry_price - stop_loss)
        reward = risk * self.tp_rr_ratio

        if direction == "BUY":
            return entry_price + reward
        else:
            return entry_price - reward

    def calculate_trailing_stop(
        self,
        direction: str,
        current_price: float,
        current_sl: float,
        atr: float,
    ) -> float:
        """Calculate trailing stop-loss level.

        Args:
            direction: Position direction.
            current_price: Current market price.
            current_sl: Current stop-loss level.
            atr: Current ATR value.

        Returns:
            New stop-loss level (only moves in favorable direction).
        """
        if not self.trailing_stop:
            return current_sl

        trail_distance = atr * self.trailing_atr_mult

        if direction == "BUY":
            new_sl = current_price - trail_distance
            return max(new_sl, current_sl)  # Only move up
        else:
            new_sl = current_price + trail_distance
            return min(new_sl, current_sl)  # Only move down

    # --------------------------------------------------
    # State Tracking
    # --------------------------------------------------

    def record_trade_result(
        self, pnl: float, pattern: str = "", instrument: str = ""
    ) -> None:
        """Record a trade result for risk tracking and adaptive analysis.

        Args:
            pnl: Profit/loss from the trade.
            pattern: Pattern type that generated the signal.
            instrument: Currency pair.
        """
        self.state.daily_pnl += pnl
        self.state.weekly_pnl += pnl
        self.state.daily_trade_count += 1

        # Feed into adaptive performance tracker
        self.adaptive.tracker.record(pnl, pattern, instrument)

        logger.info(
            f"Trade recorded: PnL={pnl:+.2f}, "
            f"Daily PnL={self.state.daily_pnl:+.2f}, "
            f"Trades today={self.state.daily_trade_count}, "
            f"Streak: {'W' if pnl > 0 else 'L'}"
            f"{self.adaptive.tracker.consecutive_wins if pnl > 0 else self.adaptive.tracker.consecutive_losses}"
        )

    def update_open_positions(self, count: int) -> None:
        """Update the count of open positions.

        Args:
            count: Current number of open positions.
        """
        self.state.open_position_count = count

    def _check_daily_reset(self) -> None:
        """Reset daily counters if a new day."""
        today = datetime.now().strftime("%Y-%m-%d")
        if self.state.last_reset_date == "":
            # First call - set date but don't reset counters
            self.state.last_reset_date = today
        elif self.state.last_reset_date != today:
            self.state.daily_pnl = 0.0
            self.state.daily_trade_count = 0
            self.state.last_reset_date = today
            logger.debug("Daily risk counters reset")

    def _check_weekly_reset(self) -> None:
        """Reset weekly counters if a new week."""
        today = datetime.now()
        week_key = f"{today.isocalendar()[0]}-W{today.isocalendar()[1]}"
        if self.state.last_weekly_reset == "":
            # First call - set week but don't reset counters
            self.state.last_weekly_reset = week_key
        elif self.state.last_weekly_reset != week_key:
            self.state.weekly_pnl = 0.0
            self.state.last_weekly_reset = week_key
            logger.debug("Weekly risk counters reset")

    def get_risk_summary(self) -> dict[str, Any]:
        """Get current risk state summary.

        Returns:
            Dictionary of risk metrics.
        """
        return {
            "trading_allowed": self.state.is_trading_allowed,
            "halt_reason": self.state.halt_reason,
            "daily_pnl": self.state.daily_pnl,
            "weekly_pnl": self.state.weekly_pnl,
            "drawdown_pct": self.state.current_drawdown_pct,
            "peak_balance": self.state.peak_balance,
            "daily_trades": self.state.daily_trade_count,
            "open_positions": self.state.open_position_count,
            "max_daily_loss_pct": self.max_daily_loss_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
            "adaptive": self.adaptive.get_summary(),
        }
