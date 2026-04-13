"""Send current status report to LINE."""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.utils.notifier import Notifier
from src.utils.config_loader import TradingConfig, create_broker


def main() -> None:
    config = TradingConfig()
    broker = create_broker(config.env)
    broker.connect()

    n = Notifier(
        line_config={
            "channel_access_token": os.getenv("LINE_CHANNEL_ACCESS_TOKEN", ""),
            "user_id": os.getenv("LINE_USER_ID", ""),
            "imgbb_api_key": os.getenv("IMGBB_API_KEY", ""),
        },
    )

    # Parse trade log
    opens: dict[str, dict] = {}
    trade_log = Path("logs/trades.log")
    if trade_log.exists():
        with open(trade_log) as f:
            for line in f:
                if " | " not in line:
                    continue
                _, _, t = line.partition(" | ")
                t = t.strip()
                if t.startswith("OPEN|"):
                    parts = t.split("|")
                    fields = {}
                    for p in parts[3:]:
                        if "=" in p:
                            k, v = p.split("=", 1)
                            fields[k] = v
                    inst = parts[1].replace("/", "_")
                    opens[inst] = {
                        "pair": parts[1],
                        "dir": parts[2],
                        "entry": float(fields.get("entry", 0)),
                        "sl": float(fields.get("sl", 0)),
                        "tp": float(fields.get("tp", 0)),
                        "size": float(fields.get("size", 0)),
                        "pattern": fields.get("pattern", ""),
                    }
                elif t.startswith("CLOSE|"):
                    inst = t.split("|")[1].replace("/", "_")
                    opens.pop(inst, None)

    # Build report
    lines = ["AI FX Bot Status Report"]
    lines.append("=" * 30)

    total_pnl = 0.0
    for inst, pos in opens.items():
        df = broker.get_historical_prices(inst, "HOUR", num_points=1)
        price = float(df["close"].iloc[-1]) if not df.empty else 0
        diff = (price - pos["entry"]) if pos["dir"] == "BUY" else (pos["entry"] - price)
        is_jpy = "JPY" in inst
        pips = diff * (100 if is_jpy else 10000)
        pnl = diff * pos["size"] * (100 if is_jpy else 10000)
        total_pnl += pnl
        icon = "+" if pnl >= 0 else ""
        dec = 3 if is_jpy else 5

        lines.append("")
        lines.append(f"{pos['pair']} {pos['dir']} {pos['size']:.2f}lots")
        lines.append(f"  Entry: {pos['entry']:.{dec}f}")
        lines.append(f"  Now:   {price:.{dec}f}")
        lines.append(f"  SL:    {pos['sl']:.{dec}f}")
        lines.append(f"  TP:    {pos['tp']:.{dec}f}")
        lines.append(f"  P&L:   {icon}{pips:.1f} pips ({icon}{pnl:,.0f})")
        lines.append(f"  Pattern: {pos['pattern']}")

    lines.append("")
    lines.append("=" * 30)
    icon_total = "+" if total_pnl >= 0 else ""
    lines.append(f"Total Unrealized: {icon_total}{total_pnl:,.0f}")
    lines.append(f"Positions: {len(opens)}")

    msg = "\n".join(lines)
    n.send("Status Report", msg)
    print(msg)
    print("\nSent to LINE!")


if __name__ == "__main__":
    main()
