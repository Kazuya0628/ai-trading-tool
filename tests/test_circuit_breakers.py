"""Tests for Gemini/LINE circuit breaker behavior."""

from __future__ import annotations

from src.strategies.ai_analyzer import AIChartAnalyzer
from src.utils.notifier import Notifier


class _DummyResponse:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400 and self.status_code != 429:
            raise RuntimeError(f"HTTP {self.status_code}")


def test_gemini_rate_limit_trip_and_daily_reset() -> None:
    analyzer = AIChartAnalyzer(api_key="")

    assert analyzer._handle_rate_limit(Exception("429 RESOURCE_EXHAUSTED"))
    assert analyzer._is_rate_limited_today()

    analyzer._rate_limited_utc_date = "2000-01-01"
    assert not analyzer._is_rate_limited_today()


def test_gemini_non_rate_limit_error_does_not_trip() -> None:
    analyzer = AIChartAnalyzer(api_key="")

    assert not analyzer._handle_rate_limit(Exception("timeout while connecting"))
    assert not analyzer._is_rate_limited_today()


def test_line_messaging_429_trips_monthly_breaker(monkeypatch) -> None:
    calls: list[str] = []

    def _fake_post(url: str, **kwargs):
        calls.append(url)
        return _DummyResponse(429)

    monkeypatch.setattr("src.utils.notifier.requests.post", _fake_post)

    notifier = Notifier(
        line_config={
            "channel_access_token": "test-token",
            "user_id": "test-user",
        }
    )

    notifier._send_line_messaging("title", "message")
    assert notifier._line_quota_exhausted()

    call_count_after_trip = len(calls)
    notifier._send_line_messaging("title2", "message2")
    assert len(calls) == call_count_after_trip


def test_line_notify_429_trips_monthly_breaker(monkeypatch) -> None:
    calls: list[str] = []

    def _fake_post(url: str, **kwargs):
        calls.append(url)
        return _DummyResponse(429)

    monkeypatch.setattr("src.utils.notifier.requests.post", _fake_post)

    notifier = Notifier(line_token="legacy-token")

    notifier._send_line_notify("title", "message")
    assert notifier._line_quota_exhausted()

    call_count_after_trip = len(calls)
    notifier._send_line_notify("title2", "message2")
    assert len(calls) == call_count_after_trip
