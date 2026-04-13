"""Web Dashboard entry point.

Re-exports the existing dashboard from src/dashboard.py and registers
the new API blueprint for extended endpoints.

Can be run standalone:
    python -m src.web.dashboard
    python src/web/dashboard.py --port 8080

Services are injected via create_app():
    app = create_app(
        position_store=...,
        portfolio_service=...,
        broker=...,
        runtime_state=...,
    )
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.dashboard import app as dashboard_app
from src.web.routes import api, configure


def create_app(
    position_store: Any = None,
    portfolio_service: Any = None,
    broker: Any = None,
    runtime_state: Any = None,
):
    """Create the Flask app with API routes and injected services.

    Args:
        position_store: PositionStore instance for trade data.
        portfolio_service: PortfolioService instance for NAV/drawdown.
        broker: Broker client for live account/position data.
        runtime_state: RuntimeState instance for system status.

    Returns:
        Configured Flask application.
    """
    configure(
        position_store=position_store,
        portfolio_service=portfolio_service,
        broker=broker,
        runtime_state=runtime_state,
    )
    # Avoid double-registration if called multiple times
    if "api" not in dashboard_app.blueprints:
        dashboard_app.register_blueprint(api)
    return dashboard_app


def main() -> None:
    parser = argparse.ArgumentParser(description="AI FX Trading Dashboard")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()

    app = create_app()
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
