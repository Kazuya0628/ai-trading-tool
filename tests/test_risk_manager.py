"""Tests for risk management module."""

import pytest

from src.models.risk_manager import RiskManager, PositionSize
from src.strategies.pattern_detector import PatternType, SignalDirection
from src.strategies.strategy_engine import TradeSignal


class TestRiskManager:
    """Test suite for RiskManager."""

    def setup_method(self) -> None:
        """Setup test fixtures."""
        self.rm = RiskManager({
            "risk_per_trade_pct": 1.0,
            "max_position_size_lots": 10.0,
            "max_open_positions": 3,
            "max_positions_per_pair": 1,
            "stop_loss_atr_multiple": 2.0,
            "min_stop_loss_pips": 10,
            "max_stop_loss_pips": 100,
            "take_profit_rr_ratio": 2.0,
            "trailing_stop_enabled": True,
            "trailing_stop_atr_multiple": 1.5,
            "max_daily_loss_pct": 2.0,
            "max_daily_trades": 10,
            "max_weekly_loss_pct": 5.0,
            "max_drawdown_pct": 10.0,
        })

    def _make_signal(
        self,
        direction: str = "BUY",
        entry: float = 150.0,
        sl: float = 149.0,
        tp: float = 152.0,
        confidence: float = 70.0,
    ) -> TradeSignal:
        """Create a test trade signal."""
        return TradeSignal(
            epic="CS.D.USDJPY.TODAY.IP",
            pair_name="USD/JPY",
            direction=SignalDirection.BUY if direction == "BUY" else SignalDirection.SELL,
            entry_price=entry,
            stop_loss=sl,
            take_profit=tp,
            confidence=confidence,
            pattern=PatternType.DOUBLE_BOTTOM,
            timeframe="HOUR_4",
        )

    # --------------------------------------------------
    # Position Sizing Tests
    # --------------------------------------------------

    def test_position_sizing_basic(self) -> None:
        """Test basic position size calculation."""
        signal = self._make_signal(entry=150.0, sl=149.0, tp=152.0)
        pos = self.rm.calculate_position_size(
            signal=signal,
            account_balance=1_000_000,
            pip_value=0.01,
        )
        assert pos.lots > 0
        assert pos.lots <= 10.0
        assert pos.risk_amount == 10_000  # 1% of 1M
        assert pos.risk_pct == 1.0

    def test_position_sizing_small_account(self) -> None:
        """Test position sizing with small account."""
        signal = self._make_signal(entry=150.0, sl=149.0, tp=152.0)
        pos = self.rm.calculate_position_size(
            signal=signal,
            account_balance=100_000,
            pip_value=0.01,
        )
        assert pos.lots > 0
        assert pos.risk_amount == 1_000  # 1% of 100K

    def test_position_sizing_caps_at_max(self) -> None:
        """Test that position size is capped at maximum."""
        signal = self._make_signal(entry=150.0, sl=149.999, tp=152.0)
        pos = self.rm.calculate_position_size(
            signal=signal,
            account_balance=100_000_000,  # Very large balance
            pip_value=0.01,
        )
        assert pos.lots <= 10.0  # Max lot cap

    # --------------------------------------------------
    # Trading Permission Tests
    # --------------------------------------------------

    def test_can_trade_normal(self) -> None:
        """Test normal trading allowed."""
        can, reason = self.rm.can_trade(1_000_000)
        assert can
        assert reason == "OK"

    def test_daily_loss_limit(self) -> None:
        """Test daily loss limit enforcement."""
        self.rm.record_trade_result(-25_000)  # 2.5% loss
        can, reason = self.rm.can_trade(1_000_000)
        assert not can
        assert "Daily loss" in reason

    def test_max_daily_trades(self) -> None:
        """Test daily trade count limit."""
        for _ in range(10):
            self.rm.record_trade_result(100)  # Small wins
        can, reason = self.rm.can_trade(1_000_000)
        assert not can
        assert "trade limit" in reason

    def test_max_open_positions(self) -> None:
        """Test max open positions limit."""
        self.rm.update_open_positions(3)
        can, reason = self.rm.can_trade(1_000_000)
        assert not can
        assert "Max open positions" in reason

    def test_drawdown_protection(self) -> None:
        """Test max drawdown halt."""
        self.rm.state.peak_balance = 1_000_000
        can, reason = self.rm.can_trade(890_000)  # 11% drawdown
        assert not can
        assert "drawdown" in reason.lower()

    # --------------------------------------------------
    # Signal Validation Tests
    # --------------------------------------------------

    def test_validate_good_signal(self) -> None:
        """Test validation of a good signal."""
        signal = self._make_signal(confidence=70.0)
        valid, reason = self.rm.validate_signal(signal)
        assert valid
        assert reason == "OK"

    def test_reject_low_confidence(self) -> None:
        """Test rejection of low confidence signal."""
        signal = self._make_signal(confidence=40.0)
        valid, reason = self.rm.validate_signal(signal)
        assert not valid
        assert "confidence" in reason.lower()

    def test_reject_no_stop_loss(self) -> None:
        """Test rejection of signal without stop-loss."""
        signal = self._make_signal(sl=0)
        valid, reason = self.rm.validate_signal(signal)
        assert not valid

    def test_reject_bad_rr(self) -> None:
        """Test rejection of poor risk-reward."""
        signal = self._make_signal(entry=150.0, sl=149.0, tp=150.5)
        valid, reason = self.rm.validate_signal(signal)
        assert not valid
        assert "R:R" in reason

    # --------------------------------------------------
    # Stop Loss Calculation Tests
    # --------------------------------------------------

    def test_atr_stop_loss_buy(self) -> None:
        """Test ATR-based stop-loss for BUY."""
        sl = self.rm.calculate_stop_loss("BUY", 150.0, atr=0.5)
        assert sl < 150.0
        assert sl == 150.0 - (0.5 * 2.0)  # ATR * multiplier

    def test_atr_stop_loss_sell(self) -> None:
        """Test ATR-based stop-loss for SELL."""
        sl = self.rm.calculate_stop_loss("SELL", 150.0, atr=0.5)
        assert sl > 150.0
        assert sl == 150.0 + (0.5 * 2.0)

    def test_take_profit_rr_ratio(self) -> None:
        """Test take-profit based on R:R ratio."""
        tp = self.rm.calculate_take_profit("BUY", 150.0, 149.0)
        assert tp == 152.0  # 2:1 R:R

        tp = self.rm.calculate_take_profit("SELL", 150.0, 151.0)
        assert tp == 148.0  # 2:1 R:R

    # --------------------------------------------------
    # Trailing Stop Tests
    # --------------------------------------------------

    def test_trailing_stop_buy(self) -> None:
        """Test trailing stop for BUY position."""
        # Price moved up, trailing stop should move up
        new_sl = self.rm.calculate_trailing_stop("BUY", 152.0, 149.0, atr=0.5)
        assert new_sl > 149.0  # Moved up
        assert new_sl == 152.0 - (0.5 * 1.5)  # ATR * trailing mult

    def test_trailing_stop_no_backward(self) -> None:
        """Test that trailing stop never moves backward."""
        # Price moved down, stop should stay
        new_sl = self.rm.calculate_trailing_stop("BUY", 148.0, 149.0, atr=0.5)
        assert new_sl == 149.0  # Should not move down

    # --------------------------------------------------
    # State Tracking Tests
    # --------------------------------------------------

    def test_record_trade(self) -> None:
        """Test trade result recording."""
        self.rm.record_trade_result(5000)
        assert self.rm.state.daily_pnl == 5000
        assert self.rm.state.daily_trade_count == 1

        self.rm.record_trade_result(-3000)
        assert self.rm.state.daily_pnl == 2000
        assert self.rm.state.daily_trade_count == 2

    def test_risk_summary(self) -> None:
        """Test risk state summary."""
        self.rm.record_trade_result(1000)
        summary = self.rm.get_risk_summary()
        assert "trading_allowed" in summary
        assert "daily_pnl" in summary
        assert summary["daily_pnl"] == 1000
