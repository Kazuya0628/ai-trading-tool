"""Notification system for trade alerts."""

import json
from typing import Any

import requests
from loguru import logger


class Notifier:
    """Send notifications for trade events via Discord / LINE."""

    def __init__(self, discord_webhook: str = "", line_token: str = "") -> None:
        self.discord_webhook = discord_webhook
        self.line_token = line_token

    def send(self, title: str, message: str, fields: dict[str, Any] | None = None) -> None:
        """Send notification to all configured channels.

        Args:
            title: Notification title.
            message: Notification body.
            fields: Additional key-value data.
        """
        if self.discord_webhook:
            self._send_discord(title, message, fields)
        if self.line_token:
            self._send_line(title, message)

    def _send_discord(self, title: str, message: str, fields: dict[str, Any] | None = None) -> None:
        """Send Discord webhook notification."""
        embed = {
            "title": f"🤖 {title}",
            "description": message,
            "color": 0x00FF00 if "BUY" in title.upper() else 0xFF0000 if "SELL" in title.upper() else 0x0099FF,
        }
        if fields:
            embed["fields"] = [
                {"name": k, "value": str(v), "inline": True}
                for k, v in fields.items()
            ]

        payload = {"embeds": [embed]}
        try:
            resp = requests.post(
                self.discord_webhook,
                json=payload,
                timeout=10,
            )
            resp.raise_for_status()
        except Exception as e:
            logger.error(f"Discord notification failed: {e}")

    def _send_line(self, title: str, message: str) -> None:
        """Send LINE Notify notification."""
        try:
            resp = requests.post(
                "https://notify-api.line.me/api/notify",
                headers={"Authorization": f"Bearer {self.line_token}"},
                data={"message": f"\n{title}\n{message}"},
                timeout=10,
            )
            resp.raise_for_status()
        except Exception as e:
            logger.error(f"LINE notification failed: {e}")

    def trade_opened(self, pair: str, direction: str, size: float, price: float,
                     sl: float, tp: float, pattern: str) -> None:
        """Notify about a new trade opening."""
        self.send(
            title=f"{direction} {pair}",
            message=f"Pattern: {pattern}",
            fields={
                "Price": f"{price:.5f}",
                "Size": f"{size:.2f}",
                "Stop Loss": f"{sl:.5f}",
                "Take Profit": f"{tp:.5f}",
            },
        )

    def trade_closed(self, pair: str, direction: str, pnl: float, pips: float) -> None:
        """Notify about a trade closing."""
        emoji = "✅" if pnl >= 0 else "❌"
        self.send(
            title=f"{emoji} Closed {direction} {pair}",
            message=f"P&L: ¥{pnl:,.0f} ({pips:+.1f} pips)",
        )

    def alert(self, message: str) -> None:
        """Send a general alert."""
        self.send(title="⚠️ Alert", message=message)
