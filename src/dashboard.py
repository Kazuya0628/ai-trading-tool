"""Real-time Web Dashboard for AI FX Trading Bot.

Lightweight Flask app displaying:
- Live candlestick charts (auto-updating)
- Open positions with entry/SL/TP levels
- Adaptive risk status
- Trade history
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template_string
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.indicators import TechnicalAnalyzer
from src.utils.config_loader import TradingConfig, create_broker

app = Flask(__name__)
config: TradingConfig | None = None
broker: Any = None
tech_analyzer: TechnicalAnalyzer | None = None

# Cache to avoid hitting Twelve Data rate limits
_cache: dict[str, Any] = {}
_cache_time: float = 0
_CACHE_TTL = 60  # seconds

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI FX Trading Bot - Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/lightweight-charts@4.1.0/dist/lightweight-charts.standalone.production.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #0a0a0f;
            color: #e0e0e0;
        }
        .header {
            background: #12121a;
            padding: 12px 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #2a2a3a;
        }
        .header h1 { font-size: 18px; color: #fff; }
        .header .status {
            display: flex;
            gap: 16px;
            font-size: 13px;
        }
        .status-dot {
            width: 8px; height: 8px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 4px;
            background: #22c55e;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.4; }
        }
        .charts-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 1px;
            background: #1a1a2a;
        }
        .chart-panel {
            background: #0f0f18;
            padding: 12px;
        }
        .chart-title {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
            font-size: 14px;
        }
        .chart-title .pair { font-weight: 700; font-size: 16px; }
        .chart-title .price { font-family: monospace; font-size: 15px; }
        .price-up { color: #22c55e; }
        .price-down { color: #ef4444; }
        .chart-container { width: 100%; height: 350px; }
        .info-bar {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1px;
            background: #1a1a2a;
            margin-top: 1px;
        }
        .info-card {
            background: #0f0f18;
            padding: 16px;
        }
        .info-card h3 {
            font-size: 12px;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 8px;
        }
        .info-card .value {
            font-size: 20px;
            font-weight: 700;
            font-family: monospace;
        }
        .positions-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }
        .positions-table th {
            text-align: left;
            padding: 8px;
            color: #888;
            font-weight: 500;
            border-bottom: 1px solid #2a2a3a;
        }
        .positions-table td {
            padding: 8px;
            border-bottom: 1px solid #1a1a2a;
            font-family: monospace;
        }
        .buy-badge {
            background: rgba(34,197,94,0.15);
            color: #22c55e;
            padding: 2px 8px;
            border-radius: 4px;
            font-weight: 600;
        }
        .sell-badge {
            background: rgba(239,68,68,0.15);
            color: #ef4444;
            padding: 2px 8px;
            border-radius: 4px;
            font-weight: 600;
        }
        .regime-badge {
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 600;
        }
        .regime-trending { background: rgba(34,197,94,0.15); color: #22c55e; }
        .regime-ranging { background: rgba(234,179,8,0.15); color: #eab308; }
        .regime-volatile { background: rgba(239,68,68,0.15); color: #ef4444; }
        .regime-normal { background: rgba(59,130,246,0.15); color: #3b82f6; }
        .footer {
            text-align: center;
            padding: 8px;
            font-size: 11px;
            color: #555;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>AI FX Trading Bot</h1>
        <div class="status">
            <span><span class="status-dot"></span> Live</span>
            <span id="update-time">--</span>
            <span id="regime-badge" class="regime-badge regime-normal">--</span>
        </div>
    </div>

    <div class="info-bar" id="info-bar">
        <div class="info-card">
            <h3>Balance</h3>
            <div class="value" id="balance">--</div>
        </div>
        <div class="info-card">
            <h3>Daily P&L</h3>
            <div class="value" id="daily-pnl">--</div>
        </div>
        <div class="info-card">
            <h3>Drawdown</h3>
            <div class="value" id="drawdown">--</div>
        </div>
        <div class="info-card">
            <h3>Adaptive Risk</h3>
            <div class="value" id="risk-mult">--</div>
        </div>
        <div class="info-card">
            <h3>Win Rate</h3>
            <div class="value" id="win-rate">--</div>
        </div>
    </div>

    <div id="loading" style="text-align:center;padding:40px;color:#888;font-size:16px;">
        Loading chart data... (may take ~20s due to API rate limits)
    </div>
    <div class="charts-grid" id="charts-grid"></div>

    <div style="padding: 12px; background: #0f0f18;">
        <h3 style="font-size: 14px; margin-bottom: 8px; color: #888;">Open Positions</h3>
        <table class="positions-table">
            <thead>
                <tr>
                    <th>Pair</th><th>Direction</th><th>Lots</th><th>Entry</th>
                    <th>Current</th><th>SL</th><th>TP</th><th>P&L</th><th>Pattern</th>
                </tr>
            </thead>
            <tbody id="positions-body"></tbody>
        </table>
    </div>

    <div class="footer">AI FX Trading Bot - AI Sennin Methodology | Auto-refresh: 30s</div>

    <script>
        const charts = {};
        const candleSeries = {};
        const slLines = {};
        const tpLines = {};
        const entryLines = {};

        function createChart(containerId, pair) {
            const container = document.getElementById(containerId);
            const chart = LightweightCharts.createChart(container, {
                width: container.clientWidth,
                height: 350,
                layout: {
                    background: { color: '#0f0f18' },
                    textColor: '#888',
                },
                grid: {
                    vertLines: { color: '#1a1a2a' },
                    horzLines: { color: '#1a1a2a' },
                },
                crosshair: { mode: 0 },
                timeScale: {
                    timeVisible: true,
                    secondsVisible: false,
                },
            });

            const series = chart.addCandlestickSeries({
                upColor: '#22c55e',
                downColor: '#ef4444',
                borderUpColor: '#22c55e',
                borderDownColor: '#ef4444',
                wickUpColor: '#22c55e',
                wickDownColor: '#ef4444',
            });

            charts[pair] = chart;
            candleSeries[pair] = series;

            window.addEventListener('resize', () => {
                chart.applyOptions({ width: container.clientWidth });
            });
        }

        async function fetchData() {
            try {
                const resp = await fetch('/api/data');
                const data = await resp.json();

                if (data.error) {
                    document.getElementById('loading').textContent = 'API Error: ' + data.error;
                    document.getElementById('loading').style.color = '#ef4444';
                    return;
                }

                // Update info cards
                document.getElementById('update-time').textContent = data.timestamp;
                document.getElementById('balance').textContent = data.balance;
                document.getElementById('daily-pnl').textContent = data.daily_pnl;
                document.getElementById('daily-pnl').style.color =
                    data.daily_pnl_raw >= 0 ? '#22c55e' : '#ef4444';
                document.getElementById('drawdown').textContent = data.drawdown;
                document.getElementById('risk-mult').textContent = data.risk_multiplier;
                document.getElementById('win-rate').textContent = data.win_rate;

                const regimeBadge = document.getElementById('regime-badge');
                regimeBadge.textContent = data.regime;
                regimeBadge.className = 'regime-badge regime-' + data.regime;

                // Update charts
                const grid = document.getElementById('charts-grid');
                for (const pair of data.pairs) {
                    const containerId = 'chart-' + pair.instrument;

                    if (!charts[pair.instrument]) {
                        const panel = document.createElement('div');
                        panel.className = 'chart-panel';
                        panel.innerHTML = `
                            <div class="chart-title">
                                <span class="pair">${pair.name}</span>
                                <span class="price" id="price-${pair.instrument}">--</span>
                            </div>
                            <div class="chart-container" id="${containerId}"></div>
                        `;
                        grid.appendChild(panel);
                        createChart(containerId, pair.instrument);
                    }

                    if (pair.candles && pair.candles.length > 0) {
                        try {
                            candleSeries[pair.instrument].setData(pair.candles);
                        } catch (chartErr) {
                            console.error('Chart error for ' + pair.instrument + ':', chartErr, 'First candle:', pair.candles[0]);
                        }
                        const last = pair.candles[pair.candles.length - 1];
                        const priceEl = document.getElementById('price-' + pair.instrument);
                        priceEl.textContent = last.close.toFixed(pair.instrument.includes('JPY') ? 3 : 5);
                        priceEl.className = 'price ' + (last.close >= last.open ? 'price-up' : 'price-down');
                    }

                    // Draw position lines
                    if (pair.position) {
                        const pos = pair.position;
                        const chart = charts[pair.instrument];
                        const series = candleSeries[pair.instrument];

                        // Remove old lines
                        if (entryLines[pair.instrument]) {
                            series.removePriceLine(entryLines[pair.instrument]);
                            series.removePriceLine(slLines[pair.instrument]);
                            series.removePriceLine(tpLines[pair.instrument]);
                        }

                        const dec = pair.instrument.includes('JPY') ? 3 : 5;
                        entryLines[pair.instrument] = series.createPriceLine({
                            price: pos.entry, color: '#3b82f6',
                            lineWidth: 2, lineStyle: 0,
                            axisLabelVisible: true,
                            title: pos.direction + ' ' + pos.size.toFixed(2) + ' lots @ ' + pos.entry.toFixed(dec) + ' [' + pos.pattern + ']',
                        });
                        slLines[pair.instrument] = series.createPriceLine({
                            price: pos.sl, color: '#ef4444',
                            lineWidth: 2, lineStyle: 2,
                            axisLabelVisible: true,
                            title: 'SL ' + pos.sl.toFixed(dec),
                        });
                        tpLines[pair.instrument] = series.createPriceLine({
                            price: pos.tp, color: '#22c55e',
                            lineWidth: 2, lineStyle: 2,
                            axisLabelVisible: true,
                            title: 'TP ' + pos.tp.toFixed(dec),
                        });
                    }
                }

                // Update positions table
                const tbody = document.getElementById('positions-body');
                if (data.positions.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="9" style="text-align:center;color:#555;">No open positions</td></tr>';
                } else {
                    tbody.innerHTML = data.positions.map(p => `
                        <tr>
                            <td>${p.pair}</td>
                            <td><span class="${p.direction === 'BUY' ? 'buy-badge' : 'sell-badge'}">${p.direction}</span></td>
                            <td>${p.lots}</td>
                            <td>${p.entry}</td>
                            <td>${p.current}</td>
                            <td style="color:#ef4444">${p.sl}</td>
                            <td style="color:#22c55e">${p.tp}</td>
                            <td style="color:${p.pnl_raw >= 0 ? '#22c55e' : '#ef4444'}">${p.pnl}</td>
                            <td>${p.pattern}</td>
                        </tr>
                    `).join('');
                }

                document.getElementById('loading').style.display = 'none';

            } catch (e) {
                console.error('Fetch error:', e);
                document.getElementById('loading').textContent = 'Error: ' + e.message;
                document.getElementById('loading').style.color = '#ef4444';
                document.getElementById('update-time').textContent = 'Error';
                document.getElementById('update-time').style.color = '#ef4444';
            }
        }

        // Show loading state
        document.getElementById('update-time').textContent = 'Loading...';
        fetchData();
        setInterval(fetchData, 60000);
    </script>
</body>
</html>
"""


def init_app() -> Flask:
    """Initialize the dashboard app with trading config."""
    global config, broker, tech_analyzer

    config = TradingConfig()
    broker = create_broker(config.env)
    broker.connect()
    tech_analyzer = TechnicalAnalyzer(config.indicators)

    return app


@app.route("/")
def index() -> str:
    """Serve the dashboard HTML."""
    return render_template_string(DASHBOARD_HTML)


@app.route("/api/data")
def api_data() -> Any:
    """API endpoint returning all dashboard data."""
    global config, broker, tech_analyzer, _cache, _cache_time

    if not broker or not config:
        return jsonify({"error": "Not initialized"}), 500

    # Return cached data if fresh (avoids rate limits)
    now = time.time()
    if _cache and (now - _cache_time) < _CACHE_TTL:
        _cache["timestamp"] = datetime.now().strftime("%H:%M:%S")
        _cache["cached"] = True
        return jsonify(_cache)

    try:
        logger.info("Dashboard: fetching fresh data...")
        broker.ensure_session()
        account = broker.get_account_info()
        balance = account.get("balance", 0)

        # Read trade log for open positions
        positions = _get_positions()

        # Build pair data with candles (with delay between requests for rate limiting)
        pairs_data = []
        for i, instrument in enumerate(config.active_pairs):
            pair_config = config.pair_configs.get(instrument, {})
            pair_name = pair_config.get("name", instrument)

            # Rate limit: wait between requests
            if i > 0:
                time.sleep(8)  # Twelve Data free: 8 req/min

            # Fetch candles
            logger.info(f"Dashboard: fetching {instrument}...")
            df = broker.get_historical_prices(instrument, "HOUR", num_points=100)
            candles = []
            if not df.empty:
                seen_times: set[int] = set()
                for idx, row in df.iterrows():
                    ts = int(idx.timestamp())
                    # Avoid duplicate timestamps (Lightweight Charts requires unique ascending times)
                    if ts in seen_times:
                        continue
                    seen_times.add(ts)
                    candles.append({
                        "time": ts,
                        "open": round(float(row["open"]), 5),
                        "high": round(float(row["high"]), 5),
                        "low": round(float(row["low"]), 5),
                        "close": round(float(row["close"]), 5),
                    })
                # Ensure ascending order
                candles.sort(key=lambda c: c["time"])
                logger.info(f"Dashboard: {instrument} -> {len(candles)} candles, "
                            f"range: {candles[0]['time']} ~ {candles[-1]['time']}")
            else:
                logger.warning(f"Dashboard: {instrument} -> no data")

            # Find position for this pair
            pos_data = None
            for p in positions:
                if p.get("instrument") == instrument:
                    pos_data = {
                        "entry": p.get("entry", 0),
                        "sl": p.get("sl", 0),
                        "tp": p.get("tp", 0),
                        "direction": p.get("direction", "BUY"),
                        "size": p.get("size", 0),
                        "pattern": p.get("pattern", ""),
                    }

            pairs_data.append({
                "instrument": instrument,
                "name": pair_name,
                "candles": candles,
                "position": pos_data,
            })

        # Build positions list for table
        positions_table = []
        for p in positions:
            instrument = p.get("instrument", "")
            pair_config = config.pair_configs.get(instrument, {})
            current_price = 0
            for pd_item in pairs_data:
                if pd_item["instrument"] == instrument and pd_item["candles"]:
                    current_price = pd_item["candles"][-1]["close"]

            entry = p.get("entry", 0)
            direction = p.get("direction", "BUY")
            size = p.get("size", 0)
            pnl_pips = (current_price - entry) if direction == "BUY" else (entry - current_price)
            pip_value = pair_config.get("pip_value", 0.01)
            pnl_amount = pnl_pips * size * (10000 if pip_value < 0.01 else 100)
            decimals = 3 if "JPY" in instrument else 5

            positions_table.append({
                "pair": pair_config.get("name", instrument),
                "direction": direction,
                "lots": f"{size:.2f}",
                "entry": f"{entry:.{decimals}f}",
                "current": f"{current_price:.{decimals}f}",
                "sl": f"{p.get('sl', 0):.{decimals}f}",
                "tp": f"{p.get('tp', 0):.{decimals}f}",
                "pnl": f"{pnl_amount:+,.0f}",
                "pnl_raw": pnl_amount,
                "pattern": p.get("pattern", ""),
            })

        result = {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "balance": f"¥{balance:,.0f}",
            "daily_pnl": f"¥{account.get('profit_loss', 0):+,.0f}",
            "daily_pnl_raw": account.get("profit_loss", 0),
            "drawdown": "0.0%",
            "risk_multiplier": "×1.00",
            "win_rate": "N/A",
            "regime": "normal",
            "pairs": pairs_data,
            "positions": positions_table,
            "cached": False,
        }

        # Cache the result
        _cache = result
        _cache_time = time.time()

        logger.info("Dashboard: data ready")
        return jsonify(result)

    except Exception as e:
        logger.error(f"Dashboard API error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/api/debug")
def api_debug() -> Any:
    """Debug endpoint to check raw data."""
    if not broker or not config:
        return jsonify({"error": "Not initialized"})

    instrument = "USD_JPY"
    df = broker.get_historical_prices(instrument, "HOUR", num_points=5)
    raw = []
    if not df.empty:
        for idx, row in df.iterrows():
            raw.append({
                "datetime": str(idx),
                "timestamp": int(idx.timestamp()),
                "open": float(row["open"]),
                "close": float(row["close"]),
            })
    return jsonify({"instrument": instrument, "count": len(raw), "data": raw})


def _get_positions() -> list[dict]:
    """Read open positions from trade log.

    Log format: '2026-03-20 08:39:19.288 | OPEN|USD/JPY|SELL|deal=...|entry=...|sl=...|tp=...|size=...|pattern=...'
    """
    trade_log = Path("logs/trades.log")
    if not trade_log.exists():
        logger.warning("No trade log found at logs/trades.log")
        return []

    positions: dict[str, dict] = {}
    closed: set[str] = set()
    try:
        with open(trade_log) as f:
            for line in f:
                # Split on ' | ' first to separate timestamp from trade data
                if " | " not in line:
                    continue
                _, _, trade_part = line.partition(" | ")
                trade_part = trade_part.strip()

                if trade_part.startswith("OPEN|"):
                    parts = trade_part.split("|")
                    # parts[0]='OPEN', parts[1]='USD/JPY', parts[2]='SELL', parts[3+]=key=value
                    if len(parts) >= 7:
                        pair_name = parts[1]
                        direction = parts[2]
                        fields = {}
                        for p in parts[3:]:
                            if "=" in p:
                                k, v = p.split("=", 1)
                                fields[k] = v

                        instrument = pair_name.replace("/", "_")
                        positions[instrument] = {
                            "instrument": instrument,
                            "direction": direction,
                            "entry": float(fields.get("entry", 0)),
                            "sl": float(fields.get("sl", 0)),
                            "tp": float(fields.get("tp", 0)),
                            "size": float(fields.get("size", 0)),
                            "pattern": fields.get("pattern", ""),
                        }

                elif trade_part.startswith("CLOSE|"):
                    parts = trade_part.split("|")
                    if len(parts) >= 2:
                        instrument = parts[1].replace("/", "_")
                        closed.add(instrument)

        # Remove closed positions
        for inst in closed:
            positions.pop(inst, None)

    except Exception as e:
        logger.error(f"Error reading trade log: {e}")

    logger.info(f"Positions from log: {list(positions.keys())}")
    return list(positions.values())


def main() -> None:
    """Run the dashboard server."""
    import argparse
    parser = argparse.ArgumentParser(description="AI FX Bot Dashboard")
    parser.add_argument("--port", type=int, default=5000, help="Port (default: 5000)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host (default: 0.0.0.0)")
    args = parser.parse_args()

    init_app()
    print(f"\n  Dashboard: http://localhost:{args.port}\n")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
