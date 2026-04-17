"""Notification system for trade alerts.

Supports: Discord (Webhook), LINE Messaging API.
"""

from __future__ import annotations

import base64
from datetime import datetime
from typing import Any

import requests
from loguru import logger


class Notifier:
    """Send notifications for trade events via Discord / LINE."""

    def __init__(
        self,
        discord_webhook: str = "",
        line_token: str = "",
        line_config: dict[str, str] | None = None,
        **_kwargs: Any,  # Accept and ignore legacy kwargs (e.g., email_config)
    ) -> None:
        self.discord_webhook = discord_webhook
        # Legacy LINE Notify (deprecated)
        self.line_token = line_token

        # LINE Messaging API
        self.line_config = line_config or {}
        self.line_channel_token = self.line_config.get("channel_access_token", "")
        self.line_user_id = self.line_config.get("user_id", "")
        self.line_enabled = bool(self.line_channel_token and self.line_user_id)

        # imgbb for image hosting (needed for LINE image messages)
        self.imgbb_api_key = self.line_config.get("imgbb_api_key", "")

        if self.line_enabled:
            logger.info(f"LINE Messaging API enabled: -> {self.line_user_id[:8]}...")

        # Circuit breakers for LINE and imgbb
        self._line_suspended_until_month: str = ""
        self._imgbb_failure_count: int = 0
        self._imgbb_suspended: bool = False

    def send(
        self,
        title: str,
        message: str,
        fields: dict[str, Any] | None = None,
        *,
        line: bool = True,
    ) -> None:
        """Send notification to all configured channels.

        Args:
            title: Notification title.
            message: Notification body.
            fields: Additional key-value data.
            line: If False, skip LINE (use for high-frequency / low-priority events
                  to stay within the monthly LINE message quota).
        """
        if self.discord_webhook:
            self._send_discord(title, message, fields)
        if line:
            if self.line_enabled:
                self._send_line_messaging(title, message, fields)
            elif self.line_token:
                self._send_line_notify(title, message)

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
        if self._line_quota_exhausted():
            logger.debug(f"LINE suspended (quota): skipping '{title}'")
            return

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
            if resp.status_code == 429:
                self._suspend_line_for_month()
                return
            resp.raise_for_status()
            logger.info(f"LINE message sent: {title}")
        except Exception as e:
            logger.error(f"LINE Messaging API failed: {e}")

    def _upload_image_imgbb(self, image_bytes: bytes) -> str | None:
        """Upload image to imgbb and return the public URL.

        Args:
            image_bytes: PNG image bytes.

        Returns:
            Public HTTPS URL or None on failure.
        """
        if not self.imgbb_api_key or self._imgbb_suspended:
            return None

        try:
            b64 = base64.b64encode(image_bytes).decode("utf-8")
            resp = requests.post(
                "https://api.imgbb.com/1/upload",
                data={
                    "key": self.imgbb_api_key,
                    "image": b64,
                    "expiration": 86400,  # 24 hours
                },
                timeout=15,
            )
            if resp.status_code == 400:
                # imgbb returns 400 for both invalid key AND rate limit.
                # Only suspend for invalid key — rate limit auto-recovers.
                try:
                    body = resp.json()
                    err_msg = body.get("error", {}).get("message", "")
                except Exception:
                    err_msg = resp.text[:200]
                if "rate limit" in err_msg.lower():
                    logger.warning(f"imgbb rate limit (will retry next cycle): {err_msg}")
                    return None
                self._imgbb_suspended = True
                logger.error(
                    f"imgbb 400 (invalid API key?), suspending: {err_msg}. "
                    f"Update IMGBB_API_KEY in .env and restart."
                )
                return None
            resp.raise_for_status()
            data = resp.json()
            url = data.get("data", {}).get("url", "")
            if url:
                self._imgbb_failure_count = 0
                logger.info(f"Image uploaded to imgbb: {url[:60]}...")
            return url or None
        except Exception as e:
            self._imgbb_failure_count += 1
            if self._imgbb_failure_count >= 3:
                self._imgbb_suspended = True
                logger.error(
                    f"imgbb failed {self._imgbb_failure_count} times, suspending: {e}"
                )
            else:
                logger.error(f"imgbb upload failed: {e}")
            return None

    def send_with_chart(
        self,
        title: str,
        message: str,
        fields: dict[str, Any] | None = None,
        chart_image: bytes | None = None,
        gemini_response: str = "",
        *,
        line: bool = True,
    ) -> None:
        """Send notification with chart image and Gemini analysis.

        Used for trade entry signals where AI analysis is available.
        Sends image + text as a pair to LINE (counts as 1 message).

        Args:
            title: Notification title.
            message: Notification body.
            fields: Additional key-value data.
            chart_image: PNG chart image bytes.
            gemini_response: Raw Gemini analysis text.
            line: If False, skip LINE.
        """
        if self.discord_webhook:
            self._send_discord(title, message, fields)

        # LINE: send image + analysis text as a single push (counts as 1 message)
        if line:
            if self.line_enabled:
                self._send_line_with_chart(title, message, fields, chart_image, gemini_response)
            elif self.line_token:
                self._send_line_notify(title, message)

    def _line_quota_exhausted(self) -> bool:
        """Check whether LINE is suspended for the current month due to 429."""
        current_month = datetime.now().strftime("%Y-%m")
        return self._line_suspended_until_month == current_month

    def _suspend_line_for_month(self) -> None:
        """Mark LINE as suspended for the current calendar month."""
        self._line_suspended_until_month = datetime.now().strftime("%Y-%m")
        logger.warning(
            f"LINE quota exhausted — suspending LINE notifications until next month "
            f"({self._line_suspended_until_month})"
        )

    def _send_line_with_chart(
        self,
        title: str,
        message: str,
        fields: dict[str, Any] | None = None,
        chart_image: bytes | None = None,
        gemini_response: str = "",
    ) -> None:
        """Send LINE message with chart image and analysis text.

        Uses a single push with multiple messages (image + text).
        LINE counts this as 1 message toward the monthly quota.
        """
        if self._line_quota_exhausted():
            logger.debug(f"LINE suspended (quota): skipping '{title}'")
            return

        messages = []

        # Upload and add image if available
        if chart_image and self.imgbb_api_key:
            image_url = self._upload_image_imgbb(chart_image)
            if image_url:
                messages.append({
                    "type": "image",
                    "originalContentUrl": image_url,
                    "previewImageUrl": image_url,
                })

        # Build analysis text
        text = f"[AI FX Bot] {title}\n{message}"
        if fields:
            text += "\n" + "\n".join(f"  {k}: {v}" for k, v in fields.items())
        if gemini_response:
            # Truncate to fit LINE's 5000 char limit
            analysis = gemini_response[:800]
            text += f"\n\n[Gemini AI]\n{analysis}"

        messages.append({"type": "text", "text": text[:5000]})

        payload = {
            "to": self.line_user_id,
            "messages": messages,
        }
        try:
            resp = requests.post(
                "https://api.line.me/v2/bot/message/push",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.line_channel_token}",
                },
                json=payload,
                timeout=15,
            )
            if resp.status_code == 429:
                self._suspend_line_for_month()
                return
            resp.raise_for_status()
            logger.info(f"LINE message+chart sent: {title}")
        except Exception as e:
            logger.error(f"LINE Messaging API (chart) failed: {e}")

    def _send_line_notify(self, title: str, message: str) -> None:
        """Send LINE Notify notification (deprecated, legacy support)."""
        if self._line_quota_exhausted():
            logger.debug(f"LINE suspended (quota): skipping '{title}'")
            return

        try:
            resp = requests.post(
                "https://notify-api.line.me/api/notify",
                headers={"Authorization": f"Bearer {self.line_token}"},
                data={"message": f"\n{title}\n{message}"},
                timeout=10,
            )
            if resp.status_code == 429:
                self._suspend_line_for_month()
                return
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

    def trade_closed(
        self,
        pair: str,
        direction: str,
        pnl: float,
        pips: float,
        entry_price: float = 0.0,
        exit_price: float = 0.0,
        opened_at: str = "",
        pattern: str = "",
        risk_reward: float = 0.0,
        decimals: int | None = None,
    ) -> None:
        """Notify about a trade closing."""
        from datetime import datetime
        result = "WIN" if pnl >= 0 else "LOSS"
        emoji = "✅" if pnl >= 0 else "❌"

        # 保有時間を計算
        hold_str = ""
        if opened_at:
            try:
                opened_dt = datetime.fromisoformat(opened_at)
                hold_minutes = int((datetime.now() - opened_dt).total_seconds() / 60)
                if hold_minutes >= 60:
                    hold_str = f"{hold_minutes // 60}h{hold_minutes % 60}m"
                else:
                    hold_str = f"{hold_minutes}m"
            except ValueError:
                hold_str = ""

        if decimals is None:
            decimals = 3 if entry_price > 50 else 5
        fields: dict[str, Any] = {}
        if entry_price:
            fields["Entry"] = f"{entry_price:.{decimals}f}"
        if exit_price:
            fields["Exit"] = f"{exit_price:.{decimals}f}"
        if pips:
            fields["Pips"] = f"{pips:+.1f}"
        if pattern:
            fields["Pattern"] = pattern
        if risk_reward:
            fields["R:R"] = f"1:{risk_reward:.1f}"
        if hold_str:
            fields["Hold"] = hold_str

        self.send(
            title=f"{emoji} CLOSED {direction} {pair} ({result})",
            message=f"P&L: {pnl:+,.0f}",
            fields=fields,
        )

    def drawdown_warning(self, drawdown_pct: float, threshold_pct: float, balance: float) -> None:
        """ドローダウン警告通知。"""
        self.send(
            title="⚠️ ドローダウン警告",
            message=f"現在のドローダウンが{threshold_pct:.0f}%を超えました",
            fields={
                "現在DD": f"{drawdown_pct:.1f}%",
                "残高": f"¥{balance:,.0f}",
                "対応": "ポジションサイズを自動縮小中",
            },
        )

    def regime_changed(
        self,
        pair: str,
        old_regime: str,
        new_regime: str,
        volatility: str,
        trend: str,
        risk_multiplier: float,
        reasoning: str,
    ) -> None:
        """相場レジーム変化通知。LINEはスキップ（頻度が高いためメール/Discordのみ）。"""
        self.send(
            title=f"📊 レジーム変化: {pair}",
            message=f"{old_regime} → {new_regime}",
            fields={
                "ボラティリティ": volatility,
                "トレンド強度": trend,
                "リスク倍率": f"×{risk_multiplier:.2f}",
                "判断理由": reasoning[:80] if reasoning else "-",
            },
            line=False,
        )

    def monthly_oos_result(self, result_text: str) -> None:
        """月次IS/OOS分析結果通知。"""
        self.send(
            title="📈 月次IS/OOS分析完了",
            message=result_text[:500],
        )

    def sl_updated(self, pair: str, direction: str, old_sl: float, new_sl: float,
                   entry: float, decimals: int | None = None) -> None:
        """Notify about a trailing stop update. LINEはスキップ（5分毎に発火するため）。"""
        if decimals is None:
            decimals = 3 if entry > 50 else 5
        self.send(
            title=f"SL Updated: {pair}",
            message=f"{direction} position trailing stop moved",
            fields={
                "Entry": f"{entry:.{decimals}f}",
                "Old SL": f"{old_sl:.{decimals}f}",
                "New SL": f"{new_sl:.{decimals}f}",
            },
            line=False,
        )

    def tp_updated(self, pair: str, direction: str, old_tp: float, new_tp: float,
                   entry: float, decimals: int | None = None) -> None:
        """Notify about a take profit update. LINEはスキップ（頻度が高いため）。"""
        if decimals is None:
            decimals = 3 if entry > 50 else 5
        self.send(
            title=f"TP Updated: {pair}",
            message=f"{direction} position take profit changed",
            fields={
                "Entry": f"{entry:.{decimals}f}",
                "Old TP": f"{old_tp:.{decimals}f}",
                "New TP": f"{new_tp:.{decimals}f}",
            },
            line=False,
        )

    def alert(self, message: str, *, line: bool = True) -> None:
        """Send a general alert."""
        self.send(title="ALERT", message=message, line=line)

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
