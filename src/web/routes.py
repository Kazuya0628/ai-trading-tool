"""API routes for the dashboard.

Design doc endpoints:
  GET /              — Dashboard HTML
  GET /api/summary   — Account summary + portfolio state
  GET /api/positions — Open positions (from broker + PositionStore)
  GET /api/history   — Trade history (from PositionStore)
  GET /api/signals/recent — Recent signals from cycle_log
  GET /api/system/status  — System status (mode, phase, uptime)
  GET /api/data      — Aggregated response (backward compat)

Services are injected via configure() before the Flask app starts.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from flask import Blueprint, jsonify
from loguru import logger

api = Blueprint("api", __name__)

# -----------------------------------------------------------------------
# Service registry — populated by configure() at startup
# -----------------------------------------------------------------------
_position_store: Any = None
_portfolio_service: Any = None
_broker: Any = None
_runtime_state: Any = None


def configure(
    position_store: Any = None,
    portfolio_service: Any = None,
    broker: Any = None,
    runtime_state: Any = None,
) -> None:
    """Inject service dependencies for API routes.

    Args:
        position_store: PositionStore instance.
        portfolio_service: PortfolioService instance.
        broker: Broker client instance.
        runtime_state: RuntimeState instance.
    """
    global _position_store, _portfolio_service, _broker, _runtime_state
    _position_store = position_store
    _portfolio_service = portfolio_service
    _broker = broker
    _runtime_state = runtime_state


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _read_ai_log() -> dict[str, Any]:
    """Read the latest AI analysis log."""
    log_path = Path("data/ai_analysis_log.json")
    if not log_path.exists():
        return {}
    try:
        with open(log_path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


# -----------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------

@api.route("/api/summary")
def api_summary():
    """Account summary + portfolio state."""
    ai_log = _read_ai_log()
    regime = ai_log.get("regime", {})

    portfolio_snapshot: dict[str, Any] = {}
    if _portfolio_service is not None:
        try:
            portfolio_snapshot = _portfolio_service.get_snapshot()
        except Exception as e:
            logger.warning(f"[Dashboard] portfolio snapshot error: {e}")

    account_info: dict[str, Any] = {}
    if _broker is not None:
        try:
            account_info = _broker.get_account_info() or {}
        except Exception as e:
            logger.warning(f"[Dashboard] broker account_info error: {e}")

    return jsonify({
        "regime": regime,
        "updated_at": ai_log.get("updated_at", ""),
        "account": {
            "balance": _safe_float(account_info.get("balance") or portfolio_snapshot.get("balance")),
            "nav": _safe_float(account_info.get("nav") or portfolio_snapshot.get("nav")),
            "unrealized_pl": _safe_float(
                account_info.get("unrealized_pl") or portfolio_snapshot.get("unrealized_pl")
            ),
            "peak_nav": _safe_float(portfolio_snapshot.get("peak_nav")),
            "drawdown_pct": _safe_float(portfolio_snapshot.get("drawdown_pct")),
            "daily_loss_pct": _safe_float(portfolio_snapshot.get("daily_loss_pct")),
            "weekly_loss_pct": _safe_float(portfolio_snapshot.get("weekly_loss_pct")),
            "daily_nav_anchor": _safe_float(portfolio_snapshot.get("daily_nav_anchor")),
            "weekly_nav_anchor": _safe_float(portfolio_snapshot.get("weekly_nav_anchor")),
            "open_positions": int(portfolio_snapshot.get("open_positions", 0)),
            "currency_exposure": portfolio_snapshot.get("currency_exposure", {}),
        },
    })


@api.route("/api/positions")
def api_positions():
    """Open positions — primary source: broker API; enriched from PositionStore."""
    positions: list[dict[str, Any]] = []

    # Primary: broker API
    if _broker is not None:
        try:
            broker_positions = _broker.get_positions() or []
            for pos in broker_positions:
                positions.append({
                    "deal_id": pos.get("deal_id", pos.get("id", "")),
                    "instrument": pos.get("instrument", pos.get("epic", "")),
                    "direction": pos.get("direction", ""),
                    "size": _safe_float(pos.get("size", pos.get("units"))),
                    "entry_price": _safe_float(pos.get("level", pos.get("price"))),
                    "current_price": _safe_float(pos.get("current_price")),
                    "stop_loss": _safe_float(pos.get("stop_loss", pos.get("stopLevel"))),
                    "take_profit": _safe_float(pos.get("take_profit", pos.get("limitLevel"))),
                    "unrealized_pl": _safe_float(pos.get("profit", pos.get("unrealized_pl"))),
                    "opened_at": pos.get("opened_at", pos.get("created_at", "")),
                })
        except Exception as e:
            logger.warning(f"[Dashboard] broker.get_positions error: {e}")

    # Enrich with PositionStore metadata (pattern, signal_id, confidence, is_fallback)
    if _position_store is not None and positions:
        try:
            store_positions = {
                p["deal_id"]: p
                for p in (_position_store.get_open_positions() or [])
            }
            for pos in positions:
                meta = store_positions.get(pos["deal_id"], {})
                pos["pattern"] = meta.get("pattern", "")
                pos["confidence"] = _safe_float(meta.get("confidence"))
                pos["is_fallback"] = bool(meta.get("is_fallback", False))
                pos["signal_id"] = meta.get("signal_id", "")
                # Use store entry_price if broker didn't provide it
                if pos["entry_price"] == 0:
                    pos["entry_price"] = _safe_float(meta.get("entry_price"))
        except Exception as e:
            logger.warning(f"[Dashboard] position_store enrichment error: {e}")

    # Fallback: PositionStore only (broker unavailable)
    if not positions and _position_store is not None:
        try:
            for pos in (_position_store.get_open_positions() or []):
                positions.append({
                    "deal_id": pos.get("deal_id", ""),
                    "instrument": pos.get("instrument", ""),
                    "direction": pos.get("direction", ""),
                    "size": _safe_float(pos.get("size")),
                    "entry_price": _safe_float(pos.get("entry_price")),
                    "current_price": 0.0,
                    "stop_loss": _safe_float(pos.get("stop_loss")),
                    "take_profit": _safe_float(pos.get("take_profit")),
                    "unrealized_pl": 0.0,
                    "opened_at": pos.get("opened_at", ""),
                    "pattern": pos.get("pattern", ""),
                    "confidence": _safe_float(pos.get("confidence")),
                    "is_fallback": bool(pos.get("is_fallback", False)),
                    "signal_id": pos.get("signal_id", ""),
                })
        except Exception as e:
            logger.warning(f"[Dashboard] position_store fallback error: {e}")

    return jsonify({"positions": positions, "count": len(positions)})


@api.route("/api/history")
def api_history():
    """Trade history from PositionStore (closed trades)."""
    history: list[dict[str, Any]] = []

    if _position_store is not None:
        try:
            for trade in (_position_store.get_trade_history(limit=50) or []):
                history.append({
                    "deal_id": trade.get("deal_id", ""),
                    "instrument": trade.get("instrument", ""),
                    "direction": trade.get("direction", ""),
                    "size": _safe_float(trade.get("size")),
                    "entry_price": _safe_float(trade.get("entry_price")),
                    "exit_price": _safe_float(trade.get("exit_price")),
                    "pnl": _safe_float(trade.get("pnl")),
                    "exit_reason": trade.get("exit_reason", ""),
                    "opened_at": trade.get("opened_at", ""),
                    "closed_at": trade.get("closed_at", ""),
                    "pattern": trade.get("pattern", ""),
                    "confidence": _safe_float(trade.get("confidence")),
                    "is_fallback": bool(trade.get("is_fallback", False)),
                })
        except Exception as e:
            logger.warning(f"[Dashboard] trade history error: {e}")

    return jsonify({"history": history, "count": len(history)})


@api.route("/api/signals/recent")
def api_signals_recent():
    """Recent signals from cycle_log."""
    ai_log = _read_ai_log()
    pairs = ai_log.get("pairs", {})

    # Enrich with cycle_log from runtime state if available
    if _runtime_state is not None:
        try:
            cycle_log = getattr(_runtime_state, "cycle_log", {})
            if cycle_log:
                pairs = cycle_log
        except Exception:
            pass

    return jsonify({"pairs": pairs})


@api.route("/api/system/status")
def api_system_status():
    """System status endpoint."""
    ai_log = _read_ai_log()

    status: dict[str, Any] = {
        "pid": ai_log.get("pid", 0),
        "updated_at": ai_log.get("updated_at", ""),
        "alive": bool(ai_log.get("updated_at")),
        "mode": "unknown",
        "phase": 1,
        "safe_stop_reason": None,
        "next_4h_time": "",
        "last_cycle_finished_at": "",
    }

    if _runtime_state is not None:
        try:
            run_mode = getattr(_runtime_state, "run_mode", None)
            phase = getattr(_runtime_state, "phase", None)
            status["mode"] = run_mode.value if run_mode is not None else "unknown"
            status["phase"] = phase.value if phase is not None else 1
            status["safe_stop_reason"] = getattr(_runtime_state, "safe_stop_reason", None)
            status["next_4h_time"] = getattr(_runtime_state, "next_4h_time", "")
            status["last_cycle_finished_at"] = getattr(
                _runtime_state, "last_cycle_finished_at", ""
            )
            status["alive"] = getattr(_runtime_state, "running", False)
        except Exception as e:
            logger.warning(f"[Dashboard] runtime_state status error: {e}")

    return jsonify(status)


@api.route("/api/data")
def api_data():
    """Aggregated API (backward compat with old dashboard)."""
    ai_log = _read_ai_log()

    # Reuse sub-endpoints logic inline for single-request aggregation
    account: dict[str, Any] = {
        "balance": 0,
        "unrealized_pl": 0,
        "daily_realized_pl": 0,
        "nav": 0,
    }
    if _portfolio_service is not None:
        try:
            snap = _portfolio_service.get_snapshot()
            account.update({
                "balance": snap.get("balance", 0),
                "unrealized_pl": snap.get("unrealized_pl", 0),
                "daily_realized_pl": snap.get("daily_realized_pl", 0),
                "nav": snap.get("nav", 0),
                "drawdown_pct": snap.get("drawdown_pct", 0),
            })
        except Exception:
            pass

    open_positions: list[dict[str, Any]] = []
    if _broker is not None:
        try:
            open_positions = _broker.get_positions() or []
        except Exception:
            pass

    trade_history: list[dict[str, Any]] = []
    if _position_store is not None:
        try:
            trade_history = _position_store.get_trade_history(limit=20) or []
        except Exception:
            pass

    return jsonify({
        "account": account,
        "pairs": open_positions,
        "cycle_log": ai_log.get("pairs", {}),
        "trade_history": trade_history,
        "regime": ai_log.get("regime", {}),
    })
