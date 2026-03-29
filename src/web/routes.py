"""API routes for the dashboard.

Design doc endpoints:
  GET /              — Dashboard HTML
  GET /api/summary   — Account summary
  GET /api/positions — Open positions
  GET /api/history   — Trade history
  GET /api/signals/recent — Recent signals from cycle_log
  GET /api/system/status  — System status (mode, phase, uptime)
  GET /api/data      — Aggregated response (backward compat)
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from flask import Blueprint, jsonify
from loguru import logger

api = Blueprint("api", __name__)


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


@api.route("/api/summary")
def api_summary():
    """Account summary endpoint."""
    ai_log = _read_ai_log()
    regime = ai_log.get("regime", {})
    return jsonify({
        "regime": regime,
        "updated_at": ai_log.get("updated_at", ""),
    })


@api.route("/api/signals/recent")
def api_signals_recent():
    """Recent signals from cycle_log."""
    ai_log = _read_ai_log()
    pairs = ai_log.get("pairs", {})
    return jsonify({"pairs": pairs})


@api.route("/api/system/status")
def api_system_status():
    """System status endpoint."""
    ai_log = _read_ai_log()
    return jsonify({
        "pid": ai_log.get("pid", 0),
        "updated_at": ai_log.get("updated_at", ""),
        "alive": bool(ai_log.get("updated_at")),
    })


@api.route("/api/data")
def api_data():
    """Aggregated API (backward compat with old dashboard)."""
    ai_log = _read_ai_log()
    return jsonify({
        "account": {
            "balance": 0,
            "unrealized_pl": 0,
            "daily_realized_pl": 0,
            "nav": 0,
        },
        "pairs": [],
        "cycle_log": ai_log.get("pairs", {}),
        "trade_history": [],
        "regime": ai_log.get("regime", {}),
    })
