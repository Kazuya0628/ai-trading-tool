"""Web Dashboard entry point.

Re-exports the existing dashboard from src/dashboard.py and registers
the new API blueprint for extended endpoints.

Can be run standalone:
    python -m src.web.dashboard
    python src/web/dashboard.py --port 8080
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.dashboard import app as dashboard_app
from src.web.routes import api


def create_app():
    """Create the Flask app with API routes registered."""
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
