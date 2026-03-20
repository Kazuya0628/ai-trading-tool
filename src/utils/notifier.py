"""Notification system for trade alerts.

Supports: Email (SMTP), Discord (Webhook), LINE Messaging API.
"""

from __future__ import annotations

import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

import requests
from loguru import logger


class Notifier:
    """Send notifications for trade events via Email / Discord / LINE."""

    def __init__(
        self,
        discord_webhook: str = "",
        line_token: str = "",
        line_config: dict[str, str] | None = None,
        email_config: dict[str, Any] | None = None,
    ) -> None:
        self.discord_webhook = discord_webhook
        # Legacy LINE Notify (deprecated)
        self.line_token = line_token

        # LINE Messaging API
        self.line_config = line_config or {}
        self.line_channel_token = self.line_config.get("channel_access_token", "")
        self.line_user_id = self.line_config.get("user_id", "")
        self.line_enabled = bool(self.line_channel_token and self.line_user_id)

        if self.line_enabled:
            logger.info(f"LINE Messaging API enabled: -> {self.line_user_id[:8]}...")

        # Email configuration
        self.email_config = email_config or {}
        self.smtp_host = self.email_config.get("smtp_host", "")
        self.smtp_port = int(self.email_config.get("smtp_port", 587))
        self.smtp_user = self.email_config.get("smtp_user", "")
        self.smtp_password = self.email_config.get("smtp_password", "")
        self.email_from = self.email_config.get("from", self.smtp_user)
        self.email_to = self.email_config.get("to", "")
        self.email_enabled = bool(self.smtp_host and self.smtp_user and self.email_to)

        if self.email_enabled:
            logger.info(f"Email notifications enabled: -> {self.email_to}")

    def send(self, title: str, message: str, fields: dict[str, Any] | None = None) -> None:
        """Send notification to all configured channels.

        Args:
            title: Notification title.
            message: Notification body.
            fields: Additional key-value data.
        """
        if self.email_enabled:
            self._send_email(title, message, fields)
        if self.discord_webhook:
            self._send_discord(title, message, fields)
        if self.line_enabled:
            self._send_line_messaging(title, message, fields)
        elif self.line_token:
            self._send_line_notify(title, message)

    def _send_email(
        self, title: str, message: str, fields: dict[str, Any] | None = None
    ) -> None:
        """Send email notification via SMTP.

        Args:
            title: Email subject suffix.
            message: Email body text.
            fields: Additional key-value data rendered as a table.
        """
        subject = f"[AI FX Bot] {title}"

        # Build HTML body
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fields_html = ""
        if fields:
            rows = "".join(
                f'<tr><td style="padding:4px 12px;font-weight:bold;">{k}</td>'
                f'<td style="padding:4px 12px;">{v}</td></tr>'
                for k, v in fields.items()
            )
            fields_html = f'<table style="border-collapse:collapse;margin:12px 0;">{rows}</table>'

        # Color based on direction
        if "BUY" in title.upper():
            color = "#22c55e"
            direction_label = "BUY (Long)"
        elif "SELL" in title.upper():
            color = "#ef4444"
            direction_label = "SELL (Short)"
        else:
            color = "#3b82f6"
            direction_label = ""

        html = f"""
        <div style="font-family:Arial,sans-serif;max-width:500px;">
            <div style="background:{color};color:white;padding:12px 16px;border-radius:8px 8px 0 0;">
                <h2 style="margin:0;font-size:18px;">{title}</h2>
            </div>
            <div style="border:1px solid #e5e7eb;border-top:none;padding:16px;border-radius:0 0 8px 8px;">
                <p style="margin:0 0 8px 0;">{message}</p>
                {fields_html}
                <p style="color:#9ca3af;font-size:12px;margin:12px 0 0 0;">{timestamp}</p>
            </div>
        </div>
        """

        # Plain text fallback
        fields_text = ""
        if fields:
            fields_text = "\n".join(f"  {k}: {v}" for k, v in fields.items())

        plain = f"{title}\n\n{message}\n\n{fields_text}\n\n{timestamp}"

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = self.email_from
        msg["To"] = self.email_to
        msg.attach(MIMEText(plain, "plain"))
        msg.attach(MIMEText(html, "html"))

        try:
            with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=15) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.sendmail(self.email_from, self.email_to.split(","), msg.as_string())
            logger.info(f"Email sent: {subject}")
        except Exception as e:
            logger.error(f"Email notification failed: {e}")

    def _send_discord(self, title: str, message: str, fields: dict[str, Any] | None = None) -> None:
        """Send Discord webhook notification."""
        embed = {
            "title": title,
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

    def _send_line_messaging(
        self, title: str, message: str, fields: dict[str, Any] | None = None
    ) -> None:
        """Send notification via LINE Messaging API (Push Message)."""
        # Build message text
        text = f"[AI FX Bot] {title}\n{message}"
        if fields:
            text += "\n" + "\n".join(f"  {k}: {v}" for k, v in fields.items())

        payload = {
            "to": self.line_user_id,
            "messages": [{"type": "text", "text": text}],
        }
        try:
            resp = requests.post(
                "https://api.line.me/v2/bot/message/push",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.line_channel_token}",
                },
                json=payload,
                timeout=10,
            )
            resp.raise_for_status()
            logger.info(f"LINE message sent: {title}")
        except Exception as e:
            logger.error(f"LINE Messaging API failed: {e}")

    def _send_line_notify(self, title: str, message: str) -> None:
        """Send LINE Notify notification (deprecated, legacy support)."""
        try:
            resp = requests.post(
                "https://notify-api.line.me/api/notify",
                headers={"Authorization": f"Bearer {self.line_token}"},
                data={"message": f"\n{title}\n{message}"},
                timeout=10,
            )
            resp.raise_for_status()
        except Exception as e:
            logger.error(f"LINE Notify failed: {e}")

    def trade_opened(self, pair: str, direction: str, size: float, price: float,
                     sl: float, tp: float, pattern: str) -> None:
        """Notify about a new trade opening."""
        rr = abs(tp - price) / abs(price - sl) if abs(price - sl) > 0 else 0
        self.send(
            title=f"{direction} {pair}",
            message=f"Pattern: {pattern}",
            fields={
                "Entry Price": f"{price:.5f}",
                "Size": f"{size:.2f} lots",
                "Stop Loss": f"{sl:.5f}",
                "Take Profit": f"{tp:.5f}",
                "R:R": f"1:{rr:.1f}",
            },
        )

    def trade_closed(self, pair: str, direction: str, pnl: float, pips: float) -> None:
        """Notify about a trade closing."""
        result = "WIN" if pnl >= 0 else "LOSS"
        self.send(
            title=f"CLOSED {direction} {pair} ({result})",
            message=f"P&L: {pnl:+,.0f} ({pips:+.1f} pips)",
        )

    def sl_updated(self, pair: str, direction: str, old_sl: float, new_sl: float,
                   entry: float) -> None:
        """Notify about a trailing stop update."""
        decimals = 3 if "JPY" in pair else 5
        self.send(
            title=f"SL Updated: {pair}",
            message=f"{direction} position trailing stop moved",
            fields={
                "Entry": f"{entry:.{decimals}f}",
                "Old SL": f"{old_sl:.{decimals}f}",
                "New SL": f"{new_sl:.{decimals}f}",
            },
        )

    def tp_updated(self, pair: str, direction: str, old_tp: float, new_tp: float,
                   entry: float) -> None:
        """Notify about a take profit update."""
        decimals = 3 if "JPY" in pair else 5
        self.send(
            title=f"TP Updated: {pair}",
            message=f"{direction} position take profit changed",
            fields={
                "Entry": f"{entry:.{decimals}f}",
                "Old TP": f"{old_tp:.{decimals}f}",
                "New TP": f"{new_tp:.{decimals}f}",
            },
        )

    def alert(self, message: str) -> None:
        """Send a general alert."""
        self.send(title="ALERT", message=message)

    def daily_summary(self, summary: dict[str, Any]) -> None:
        """Send daily performance summary.

        Args:
            summary: Dict with daily trading stats.
        """
        self.send(
            title="Daily Summary",
            message=f"Trades: {summary.get('trades', 0)}",
            fields={
                "Daily P&L": f"{summary.get('daily_pnl', 0):+,.0f}",
                "Win Rate": f"{summary.get('win_rate', 'N/A')}",
                "Open Positions": f"{summary.get('open_positions', 0)}",
                "Drawdown": f"{summary.get('drawdown_pct', 0):.1f}%",
                "Regime": f"{summary.get('market_regime', 'N/A')}",
            },
        )
