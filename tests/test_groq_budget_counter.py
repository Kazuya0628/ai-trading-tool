"""Tests for Groq daily budget counter and ai_log budget output."""

from __future__ import annotations

import json
from types import SimpleNamespace

from src.strategies.groq_reviewer import GroqReviewer
from src.trading_bot import TradingBot


class _FakeGroqResponse:
    def __init__(self, content: str) -> None:
        self.choices = [SimpleNamespace(message=SimpleNamespace(content=content))]


class _FakeGroqCompletions:
    def __init__(self, content: str) -> None:
        self.calls = 0
        self._content = content

    def create(self, **_: object) -> _FakeGroqResponse:
        self.calls += 1
        return _FakeGroqResponse(self._content)


class _FakeGroqClient:
    def __init__(self, content: str) -> None:
        self.chat = SimpleNamespace(completions=_FakeGroqCompletions(content))


def test_groq_reviewer_counts_daily_calls_and_remaining() -> None:
    client = _FakeGroqClient(
        '{"regime":"trending","volatility_level":"high","trend_strength":"strong",'
        '"risk_multiplier":0.9,"confidence":77,"reasoning":"trend継続"}'
    )
    reviewer = GroqReviewer(client=client, daily_limit=3)

    result = reviewer.analyze_market_regime(
        indicators={},
        sentiment_data={},
        pair_name="USD/JPY",
    )

    assert result["regime"] == "trending"
    assert reviewer.daily_count == 1
    assert reviewer.daily_limit == 3
    assert reviewer.daily_remaining == 2


def test_groq_reviewer_blocks_calls_when_daily_limit_is_reached() -> None:
    client = _FakeGroqClient(
        '{"regime":"ranging","volatility_level":"medium","trend_strength":"moderate",'
        '"risk_multiplier":1.0,"confidence":50,"reasoning":"様子見"}'
    )
    reviewer = GroqReviewer(client=client, daily_limit=1)

    first = reviewer.analyze_market_regime(
        indicators={},
        sentiment_data={},
        pair_name="USD/JPY",
    )
    second = reviewer.analyze_market_regime(
        indicators={},
        sentiment_data={},
        pair_name="USD/JPY",
    )

    assert first["regime"] == "ranging"
    assert reviewer.available is False
    assert reviewer.daily_remaining == 0
    assert second["reasoning"] == "Groq未使用のためデフォルト値"
    assert client.chat.completions.calls == 1


def test_write_ai_log_includes_groq_budget_block(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    market_regime = SimpleNamespace(
        regime="trending",
        volatility_level="high",
        trend_strength="strong",
        reasoning="trend",
        risk_multiplier=0.8,
    )
    adaptive = SimpleNamespace(market_regime=market_regime)
    risk_manager = SimpleNamespace(adaptive=adaptive)
    groq_reviewer = SimpleNamespace(
        available=True,
        daily_count=4,
        daily_limit=12,
        daily_remaining=8,
    )
    bot_stub = SimpleNamespace(
        risk_manager=risk_manager,
        groq_reviewer=groq_reviewer,
        _cycle_log={"USD_JPY": {"overall": {"direction": "BUY"}}},
        _gemini_daily_count=2,
        _gemini_daily_limit=18,
    )

    TradingBot._write_ai_log(bot_stub)

    payload = json.loads((tmp_path / "data" / "ai_analysis_log.json").read_text(encoding="utf-8"))
    assert payload["budget"]["gemini"]["daily_remaining"] == 16
    assert payload["budget"]["groq"]["daily_used"] == 4
    assert payload["budget"]["groq"]["daily_hard_limit"] == 12
    assert payload["budget"]["groq"]["daily_remaining"] == 8
