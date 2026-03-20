"""Forward Test vs Backtest Divergence Checker.

Compares live paper trading performance against backtest expectations
and alerts when metrics diverge beyond acceptable thresholds.

Usage:
    python src/divergence_checker.py                  # Check and display
    python src/divergence_checker.py --email           # Check and send email report
    python src/divergence_checker.py --json            # Output JSON
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Backtest baseline (3-year Adaptive) ─────────────────────
BACKTEST_BASELINE: dict[str, dict[str, float]] = {
    "USD_JPY": {
        "win_rate": 60.8,
        "profit_factor": 2.53,
        "max_drawdown_pct": 10.8,
        "avg_rr_ratio": 2.0,
        "sharpe_ratio": 3.01,
        "max_consecutive_losses": 21,
    },
    "EUR_USD": {
        "win_rate": 68.0,
        "profit_factor": 5.34,
        "max_drawdown_pct": 6.0,
        "sharpe_ratio": 6.98,
        "avg_rr_ratio": 2.0,
        "max_consecutive_losses": 14,
    },
    "GBP_JPY": {
        "win_rate": 73.8,
        "profit_factor": 4.76,
        "max_drawdown_pct": 6.8,
        "sharpe_ratio": 6.34,
        "avg_rr_ratio": 2.0,
        "max_consecutive_losses": 20,
    },
}

# ── Alert thresholds (percentage deviation from backtest) ────
THRESHOLDS = {
    "win_rate_drop": 15.0,       # Alert if win rate drops >15pt below backtest
    "pf_ratio": 0.5,             # Alert if PF < 50% of backtest PF
    "max_dd_exceed": 1.5,        # Alert if DD exceeds 150% of backtest DD
    "consecutive_loss_ratio": 1.5,  # Alert if consecutive losses > 150% of backtest
    "min_trades_for_check": 10,  # Minimum trades before checking
}


@dataclass
class TradeRecord:
    """A single completed trade from the log."""
    timestamp: str
    instrument: str
    direction: str
    entry_price: float
    exit_price: float
    sl: float
    tp: float
    size: float
    pattern: str
    pnl: float
    pnl_pips: float
    exit_reason: str  # "sl", "tp", "manual"
    is_win: bool


@dataclass
class ForwardMetrics:
    """Aggregated forward test metrics for one instrument."""
    instrument: str
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    max_drawdown_pct: float = 0.0
    max_consecutive_losses: int = 0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    trades: list[TradeRecord] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        return (self.wins / self.total_trades * 100) if self.total_trades > 0 else 0.0

    @property
    def profit_factor(self) -> float:
        return (self.gross_profit / abs(self.gross_loss)) if self.gross_loss != 0 else float("inf")


@dataclass
class DivergenceAlert:
    """A single divergence alert."""
    instrument: str
    metric: str
    severity: str  # "WARNING", "CRITICAL"
    backtest_value: float
    forward_value: float
    deviation_pct: float
    message: str


def parse_trade_log(log_path: str = "logs/trades.log") -> dict[str, list[dict]]:
    """Parse the trade log and extract open/close pairs.

    Returns:
        Dict of instrument -> list of trade dicts.
    """
    path = Path(log_path)
    if not path.exists():
        logger.warning(f"Trade log not found: {log_path}")
        return {}

    opens: dict[str, dict] = {}
    closed_trades: dict[str, list[dict]] = {}

    with open(path) as f:
        for line in f:
            if " | " not in line:
                continue
            timestamp_part, _, trade_part = line.partition(" | ")
            trade_part = trade_part.strip()

            if trade_part.startswith("OPEN|"):
                parts = trade_part.split("|")
                if len(parts) >= 7:
                    pair_name = parts[1]
                    direction = parts[2]
                    fields = {}
                    for p in parts[3:]:
                        if "=" in p:
                            k, v = p.split("=", 1)
                            fields[k] = v

                    instrument = pair_name.replace("/", "_")
                    opens[instrument] = {
                        "timestamp": timestamp_part.strip(),
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
                    pair_name = parts[1]
                    instrument = pair_name.replace("/", "_")
                    fields = {}
                    for p in parts[2:]:
                        if "=" in p:
                            k, v = p.split("=", 1)
                            fields[k] = v

                    if instrument in opens:
                        open_trade = opens.pop(instrument)
                        exit_price = float(fields.get("exit", 0))
                        pnl = float(fields.get("pnl", 0))

                        entry = open_trade["entry"]
                        direction = open_trade["direction"]
                        is_jpy = "JPY" in instrument
                        pip_mult = 100 if is_jpy else 10000

                        if direction == "BUY":
                            pnl_pips = (exit_price - entry) * pip_mult
                        else:
                            pnl_pips = (entry - exit_price) * pip_mult

                        # Determine exit reason
                        sl = open_trade["sl"]
                        tp = open_trade["tp"]
                        if direction == "BUY":
                            exit_reason = "tp" if exit_price >= tp else ("sl" if exit_price <= sl else "manual")
                        else:
                            exit_reason = "tp" if exit_price <= tp else ("sl" if exit_price >= sl else "manual")

                        trade = {
                            "timestamp": timestamp_part.strip(),
                            "instrument": instrument,
                            "direction": direction,
                            "entry": entry,
                            "exit": exit_price,
                            "sl": sl,
                            "tp": tp,
                            "size": open_trade["size"],
                            "pattern": open_trade["pattern"],
                            "pnl": pnl,
                            "pnl_pips": pnl_pips,
                            "exit_reason": exit_reason,
                            "is_win": pnl >= 0,
                        }

                        if instrument not in closed_trades:
                            closed_trades[instrument] = []
                        closed_trades[instrument].append(trade)

    return closed_trades


def calculate_forward_metrics(trades: list[dict]) -> ForwardMetrics:
    """Calculate forward test metrics from a list of trades."""
    if not trades:
        return ForwardMetrics(instrument="")

    instrument = trades[0]["instrument"]
    metrics = ForwardMetrics(instrument=instrument)
    metrics.total_trades = len(trades)

    equity = 0.0
    peak_equity = 0.0
    max_dd = 0.0
    consecutive_losses = 0
    max_consec = 0

    for t in trades:
        pnl = t["pnl"]
        if pnl >= 0:
            metrics.wins += 1
            metrics.gross_profit += pnl
            consecutive_losses = 0
        else:
            metrics.losses += 1
            metrics.gross_loss += pnl
            consecutive_losses += 1
            max_consec = max(max_consec, consecutive_losses)

        metrics.total_pnl += pnl
        equity += pnl
        peak_equity = max(peak_equity, equity)
        dd = (peak_equity - equity) / max(peak_equity, 1) * 100
        max_dd = max(max_dd, dd)

    metrics.max_drawdown_pct = max_dd
    metrics.max_consecutive_losses = max_consec
    if metrics.wins > 0:
        metrics.avg_win = metrics.gross_profit / metrics.wins
    if metrics.losses > 0:
        metrics.avg_loss = metrics.gross_loss / metrics.losses

    return metrics


def check_divergence(
    forward: ForwardMetrics,
    baseline: dict[str, float],
    thresholds: dict[str, float] | None = None,
) -> list[DivergenceAlert]:
    """Compare forward metrics against backtest baseline.

    Returns:
        List of DivergenceAlert objects.
    """
    t = thresholds or THRESHOLDS
    alerts: list[DivergenceAlert] = []

    if forward.total_trades < t.get("min_trades_for_check", 10):
        return alerts

    inst = forward.instrument

    # 1. Win rate check
    bt_wr = baseline.get("win_rate", 0)
    fw_wr = forward.win_rate
    wr_diff = bt_wr - fw_wr
    if wr_diff > t["win_rate_drop"]:
        severity = "CRITICAL" if wr_diff > t["win_rate_drop"] * 1.5 else "WARNING"
        alerts.append(DivergenceAlert(
            instrument=inst,
            metric="Win Rate",
            severity=severity,
            backtest_value=bt_wr,
            forward_value=fw_wr,
            deviation_pct=wr_diff,
            message=f"勝率がバックテストより{wr_diff:.1f}pt低下（BT:{bt_wr:.1f}% → FW:{fw_wr:.1f}%）",
        ))

    # 2. Profit Factor check
    bt_pf = baseline.get("profit_factor", 0)
    fw_pf = forward.profit_factor
    if bt_pf > 0 and fw_pf < bt_pf * t["pf_ratio"]:
        ratio = fw_pf / bt_pf * 100
        severity = "CRITICAL" if fw_pf < 1.0 else "WARNING"
        alerts.append(DivergenceAlert(
            instrument=inst,
            metric="Profit Factor",
            severity=severity,
            backtest_value=bt_pf,
            forward_value=fw_pf,
            deviation_pct=100 - ratio,
            message=f"PFがバックテストの{ratio:.0f}%に低下（BT:{bt_pf:.2f} → FW:{fw_pf:.2f}）",
        ))

    # 3. Max Drawdown check
    bt_dd = baseline.get("max_drawdown_pct", 0)
    fw_dd = forward.max_drawdown_pct
    if bt_dd > 0 and fw_dd > bt_dd * t["max_dd_exceed"]:
        ratio = fw_dd / bt_dd
        severity = "CRITICAL" if ratio > 2.0 else "WARNING"
        alerts.append(DivergenceAlert(
            instrument=inst,
            metric="Max Drawdown",
            severity=severity,
            backtest_value=bt_dd,
            forward_value=fw_dd,
            deviation_pct=(ratio - 1) * 100,
            message=f"最大DDがバックテストの{ratio:.1f}倍に拡大（BT:{bt_dd:.1f}% → FW:{fw_dd:.1f}%）",
        ))

    # 4. Consecutive losses check
    bt_cl = baseline.get("max_consecutive_losses", 0)
    fw_cl = forward.max_consecutive_losses
    if bt_cl > 0 and fw_cl > bt_cl * t["consecutive_loss_ratio"]:
        ratio = fw_cl / bt_cl
        severity = "CRITICAL" if ratio > 2.0 else "WARNING"
        alerts.append(DivergenceAlert(
            instrument=inst,
            metric="Consecutive Losses",
            severity=severity,
            backtest_value=bt_cl,
            forward_value=fw_cl,
            deviation_pct=(ratio - 1) * 100,
            message=f"最大連敗がバックテストの{ratio:.1f}倍（BT:{bt_cl:.0f} → FW:{fw_cl}）",
        ))

    return alerts


def generate_report(
    all_metrics: dict[str, ForwardMetrics],
    all_alerts: dict[str, list[DivergenceAlert]],
) -> str:
    """Generate a human-readable divergence report."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "=" * 60,
        f"  Forward vs Backtest Divergence Report",
        f"  Generated: {now}",
        "=" * 60,
        "",
    ]

    total_alerts = sum(len(a) for a in all_alerts.values())
    if total_alerts == 0:
        lines.append("  Status: ALL CLEAR - No divergence detected")
        lines.append("")

    for inst, metrics in all_metrics.items():
        baseline = BACKTEST_BASELINE.get(inst, {})
        alerts = all_alerts.get(inst, [])

        lines.append(f"  {inst.replace('_', '/')}")
        lines.append(f"  {'-' * 50}")

        if metrics.total_trades == 0:
            lines.append("    No completed trades yet")
            lines.append("")
            continue

        lines.append(f"    {'Metric':<25} {'Backtest':>10} {'Forward':>10} {'Status':>10}")
        lines.append(f"    {'-'*25} {'-'*10} {'-'*10} {'-'*10}")

        bt_wr = baseline.get("win_rate", 0)
        status_wr = _status_icon(metrics.win_rate, bt_wr, higher_better=True, threshold=15)
        lines.append(f"    {'Win Rate':<25} {bt_wr:>9.1f}% {metrics.win_rate:>9.1f}% {status_wr:>10}")

        bt_pf = baseline.get("profit_factor", 0)
        status_pf = _status_icon(metrics.profit_factor, bt_pf, higher_better=True, threshold=50)
        lines.append(f"    {'Profit Factor':<25} {bt_pf:>10.2f} {metrics.profit_factor:>10.2f} {status_pf:>10}")

        bt_dd = baseline.get("max_drawdown_pct", 0)
        status_dd = _status_icon(metrics.max_drawdown_pct, bt_dd, higher_better=False, threshold=50)
        lines.append(f"    {'Max Drawdown':<25} {bt_dd:>9.1f}% {metrics.max_drawdown_pct:>9.1f}% {status_dd:>10}")

        bt_cl = baseline.get("max_consecutive_losses", 0)
        status_cl = _status_icon(metrics.max_consecutive_losses, bt_cl, higher_better=False, threshold=50)
        lines.append(f"    {'Max Consec. Losses':<25} {bt_cl:>10.0f} {metrics.max_consecutive_losses:>10} {status_cl:>10}")

        lines.append(f"    {'Total Trades':<25} {'--':>10} {metrics.total_trades:>10}")
        lines.append(f"    {'Total P&L':<25} {'--':>10} {metrics.total_pnl:>+10,.0f}")
        lines.append("")

        if alerts:
            for a in alerts:
                icon = "CRITICAL" if a.severity == "CRITICAL" else "WARNING"
                lines.append(f"    [{icon}] {a.message}")
            lines.append("")

    # Overall recommendation
    critical_count = sum(1 for alerts in all_alerts.values() for a in alerts if a.severity == "CRITICAL")
    warning_count = sum(1 for alerts in all_alerts.values() for a in alerts if a.severity == "WARNING")

    lines.append("=" * 60)
    if critical_count > 0:
        lines.append(f"  RECOMMENDATION: {critical_count} CRITICAL alerts detected")
        lines.append("  Consider pausing trading and reviewing strategy parameters.")
    elif warning_count > 0:
        lines.append(f"  RECOMMENDATION: {warning_count} WARNING alerts detected")
        lines.append("  Monitor closely. No immediate action required.")
    else:
        lines.append("  RECOMMENDATION: Performance within expected range.")
        lines.append("  Continue forward testing.")
    lines.append("=" * 60)

    return "\n".join(lines)


def _status_icon(forward_val: float, backtest_val: float,
                 higher_better: bool, threshold: float) -> str:
    """Return OK/WARN/CRIT status based on deviation."""
    if backtest_val == 0:
        return "N/A"

    if higher_better:
        deviation = (backtest_val - forward_val) / backtest_val * 100
    else:
        deviation = (forward_val - backtest_val) / backtest_val * 100

    if deviation < 0:
        return "OK(+)"
    elif deviation < threshold:
        return "OK"
    elif deviation < threshold * 1.5:
        return "WARN"
    else:
        return "CRIT"


def run_check(log_path: str = "logs/trades.log", send_email: bool = False) -> dict[str, Any]:
    """Run the full divergence check.

    Returns:
        Dict with metrics, alerts, and report text.
    """
    # Parse trades
    closed_trades = parse_trade_log(log_path)

    all_metrics: dict[str, ForwardMetrics] = {}
    all_alerts: dict[str, list[DivergenceAlert]] = {}

    for inst in BACKTEST_BASELINE:
        trades = closed_trades.get(inst, [])
        metrics = calculate_forward_metrics(trades)
        metrics.instrument = inst
        all_metrics[inst] = metrics

        baseline = BACKTEST_BASELINE[inst]
        alerts = check_divergence(metrics, baseline)
        all_alerts[inst] = alerts

    report = generate_report(all_metrics, all_alerts)

    # Send email if requested
    if send_email:
        _send_email_report(report, all_alerts)

    return {
        "metrics": {inst: {
            "total_trades": m.total_trades,
            "win_rate": round(m.win_rate, 1),
            "profit_factor": round(m.profit_factor, 2),
            "total_pnl": round(m.total_pnl, 2),
            "max_drawdown_pct": round(m.max_drawdown_pct, 1),
            "max_consecutive_losses": m.max_consecutive_losses,
        } for inst, m in all_metrics.items()},
        "alerts": {inst: [
            {
                "metric": a.metric,
                "severity": a.severity,
                "message": a.message,
                "backtest": a.backtest_value,
                "forward": a.forward_value,
            } for a in alerts
        ] for inst, alerts in all_alerts.items()},
        "report": report,
    }


def _send_email_report(report: str, all_alerts: dict[str, list[DivergenceAlert]]) -> None:
    """Send divergence report via email."""
    import os
    from dotenv import load_dotenv
    load_dotenv()

    from src.utils.notifier import Notifier

    email_config = {
        "smtp_host": os.getenv("SMTP_HOST", ""),
        "smtp_port": os.getenv("SMTP_PORT", "587"),
        "smtp_user": os.getenv("SMTP_USER", ""),
        "smtp_password": os.getenv("SMTP_PASSWORD", ""),
        "from": os.getenv("EMAIL_FROM", ""),
        "to": os.getenv("EMAIL_TO", ""),
    }

    notifier = Notifier(email_config=email_config)

    critical = sum(1 for alerts in all_alerts.values() for a in alerts if a.severity == "CRITICAL")
    warning = sum(1 for alerts in all_alerts.values() for a in alerts if a.severity == "WARNING")

    if critical > 0:
        title = f"DIVERGENCE ALERT: {critical} CRITICAL"
    elif warning > 0:
        title = f"Divergence Check: {warning} warnings"
    else:
        title = "Divergence Check: All Clear"

    notifier.send(title=title, message=report)


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Forward vs Backtest Divergence Checker")
    parser.add_argument("--log", default="logs/trades.log", help="Trade log path")
    parser.add_argument("--email", action="store_true", help="Send report via email")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    result = run_check(log_path=args.log, send_email=args.email)

    if args.json:
        # Remove report from JSON output (too long)
        output = {k: v for k, v in result.items() if k != "report"}
        print(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        print(result["report"])


if __name__ == "__main__":
    main()
