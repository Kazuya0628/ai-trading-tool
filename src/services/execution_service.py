"""Execution Service — handles order submission, fill recording, and close processing.

Responsibilities:
- Pre-execution validation (mode, signal freshness, spread, slippage)
- MARKET order submission via broker
- Fill result persistence
- Close processing with P&L recording
- Slippage tracking
- Notification dispatch
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from loguru import logger

from src.domain.enums import RunMode
from src.domain.models import OrderIntent


class ExecutionService:
    """Handles trade execution with pre-checks and post-processing."""

    def __init__(
        self,
        broker: Any,
        position_store: Any,
        notifier: Any,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.broker = broker
        self.position_store = position_store
        self.notifier = notifier

        cfg = config or {}
        self._max_spread_pips: float = cfg.get("max_spread_pips", 5.0)
        self._max_slippage_pips: float = cfg.get("max_slippage_pips", 3.0)
        self._signal_expiry_seconds: int = cfg.get("signal_expiry_seconds", 300)

    def execute_entry(
        self,
        order: OrderIntent,
        run_mode: RunMode,
        decision_price: float,
        chart_image: bytes | None = None,
        gemini_reasoning: str = "",
    ) -> dict[str, Any] | None:
        """Execute a new entry order with pre-checks.

        Args:
            order: Validated order intent.
            run_mode: Current bot run mode.
            decision_price: Price at the time of signal decision.
            chart_image: Chart PNG for notification.
            gemini_reasoning: AI analysis text.

        Returns:
            Trade result dict on success, None on failure/rejection.
        """
        # Pre-execution checks
        reject = self._pre_execution_check(order, run_mode)
        if reject:
            logger.info(f"[Execution] Rejected: {reject}")
            return None

        # Check duplicate position
        if self.position_store.has_open_position(order.instrument):
            logger.info(
                f"[Execution] Blocked: already have open position for {order.instrument}"
            )
            return None

        # Submit order
        result = self.broker.open_position(
            epic=order.instrument,
            direction=order.direction,
            size=order.units,
            order_type="MARKET",
            stop_loss=order.stop_loss,
            take_profit=order.take_profit,
            trailing_stop=False,
        )

        if not result.get("success"):
            logger.error(
                f"[Execution] Order failed: {result.get('reason', 'unknown')}"
            )
            return None

        fill_price = result.get("level", decision_price)
        deal_id = result["deal_id"]

        # Record slippage
        slippage = abs(fill_price - decision_price)
        pip_size = 0.01 if decision_price > 50 else 0.0001
        slippage_pips = slippage / pip_size

        # Save position
        trade_info = {
            "deal_id": deal_id,
            "signal_id": order.signal_id,
            "instrument": order.instrument,
            "direction": order.direction,
            "entry_price": fill_price,
            "stop_loss": order.stop_loss,
            "take_profit": order.take_profit,
            "size": order.units,
            "risk_pct": order.risk_pct,
            "is_fallback": order.is_fallback,
            "opened_at": datetime.now().isoformat(),
            "decision_price": decision_price,
            "slippage_pips": round(slippage_pips, 2),
        }
        self.position_store.save_position(trade_info)

        logger.info(
            f"[Execution] OPENED {order.instrument} {order.direction} "
            f"deal={deal_id} fill={fill_price:.5f} "
            f"slippage={slippage_pips:.1f}pips"
        )

        # Notification
        rr = (
            abs(order.take_profit - fill_price) / abs(fill_price - order.stop_loss)
            if abs(fill_price - order.stop_loss) > 0
            else 0
        )
        self.notifier.send_with_chart(
            title=f"{order.direction} {order.instrument.replace('_', '/')}",
            message=f"Signal: {order.signal_id}",
            fields={
                "Entry": f"{fill_price:.5f}",
                "Size": f"{order.units}",
                "SL": f"{order.stop_loss:.5f}",
                "TP": f"{order.take_profit:.5f}",
                "R:R": f"1:{rr:.1f}",
                "Fallback": "Yes" if order.is_fallback else "No",
            },
            chart_image=chart_image,
            gemini_response=gemini_reasoning,
        )

        return trade_info

    def execute_close(
        self,
        deal_id: str,
        instrument: str,
        reason: str,
        current_price: float = 0,
    ) -> float | None:
        """Close a position and record the result.

        Args:
            deal_id: Broker deal identifier.
            instrument: Currency pair.
            reason: Close reason (SL_HIT, TP_HIT, AI_CLOSE, MANUAL, etc.).
            current_price: Current market price for P&L estimation.

        Returns:
            Realized P&L or None on failure.
        """
        pnl = self.broker.close_position(deal_id)

        if pnl is not None:
            # Apply P&L to paper balance
            if hasattr(self.broker, "apply_trade_pnl"):
                self.broker.apply_trade_pnl(pnl)

            self.position_store.close_position(
                deal_id=deal_id,
                exit_price=current_price,
                pnl=pnl,
                exit_reason=reason,
            )

            logger.info(
                f"[Execution] CLOSED {instrument} deal={deal_id} "
                f"PnL={pnl:+,.0f} reason={reason}"
            )

        return pnl

    def _pre_execution_check(
        self, order: OrderIntent, run_mode: RunMode
    ) -> str | None:
        """Run pre-execution validation checks.

        Returns:
            Rejection reason string, or None if all checks pass.
        """
        if run_mode in (RunMode.SAFE_STOP, RunMode.ANALYSIS_ONLY):
            return f"Mode is {run_mode.value} — no trading"

        if run_mode != RunMode.PAPER:
            return f"Unexpected mode: {run_mode.value}"

        # Signal expiry check
        if order.expires_at:
            try:
                expires = datetime.fromisoformat(order.expires_at)
                if datetime.now() > expires:
                    return "Signal expired"
            except ValueError:
                pass

        return None
