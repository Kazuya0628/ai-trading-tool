"""Tests for process-based bot status resolution in dashboard API."""

from __future__ import annotations

from pathlib import Path

from src import dashboard


def test_is_main_bot_command_true_for_main_py() -> None:
    main_py = Path(dashboard.__file__).resolve().parent.parent / "main.py"
    assert dashboard._is_main_bot_command(f'/usr/bin/python "{main_py}"')


def test_is_main_bot_command_false_for_dashboard() -> None:
    assert not dashboard._is_main_bot_command(
        "/usr/bin/python /work/src/dashboard.py"
    )


def test_is_main_bot_command_false_for_other_project_main_py() -> None:
    assert not dashboard._is_main_bot_command("/usr/bin/python /tmp/other/main.py")


def test_is_main_bot_command_false_for_one_shot_mode() -> None:
    main_py = Path(dashboard.__file__).resolve().parent.parent / "main.py"
    assert not dashboard._is_main_bot_command(
        f'/usr/bin/python "{main_py}" --backtest'
    )


def test_resolve_bot_process_status_prefers_detected_process(monkeypatch) -> None:
    calls = {"detect": 0}

    def _fake_detect():
        calls["detect"] += 1
        return {"alive": True, "pid": 12345}

    monkeypatch.setattr(
        dashboard,
        "_detect_main_bot_process",
        _fake_detect,
    )

    called = {"count": 0}

    def _fake_is_pid_main_bot(pid: int) -> bool:
        called["count"] += 1
        return pid == 99999

    monkeypatch.setattr(dashboard, "_is_pid_main_bot", _fake_is_pid_main_bot)

    result = dashboard._resolve_bot_process_status(log_pid="invalid")

    assert result == {"process_alive": True, "process_pid": 12345}
    assert called["count"] == 0
    assert calls["detect"] == 1


def test_resolve_bot_process_status_falls_back_to_log_pid(monkeypatch) -> None:
    monkeypatch.setattr(
        dashboard,
        "_detect_main_bot_process",
        lambda: {"alive": False, "pid": None},
    )
    monkeypatch.setattr(dashboard, "_is_pid_main_bot", lambda pid: pid == 777)

    result = dashboard._resolve_bot_process_status(log_pid="777")

    assert result == {"process_alive": True, "process_pid": 777}


def test_resolve_bot_process_status_false_when_no_process(monkeypatch) -> None:
    monkeypatch.setattr(
        dashboard,
        "_detect_main_bot_process",
        lambda: {"alive": False, "pid": None},
    )
    monkeypatch.setattr(dashboard, "_is_pid_main_bot", lambda pid: False)

    result = dashboard._resolve_bot_process_status(log_pid="invalid")

    assert result == {"process_alive": False, "process_pid": None}


def test_derive_budget_snapshot_from_legacy_pairs_payload() -> None:
    payload = {
        "pairs": {
            "USD_JPY": {
                "gemini_status": {
                    "budget_used": 4,
                    "budget_limit": 18,
                },
                "groq_vote": {
                    "reasoning": "RSIが中立域のため様子見",
                },
            }
        }
    }

    snapshot = dashboard._derive_budget_snapshot(payload)

    assert snapshot["gemini"]["daily_used"] == 4
    assert snapshot["gemini"]["daily_hard_limit"] == 18
    assert snapshot["gemini"]["daily_remaining"] == 14
    assert snapshot["groq"]["available"] is True
    assert snapshot["groq"]["daily_remaining_label"] == "利用可"


def test_derive_budget_snapshot_prefers_explicit_budget_block() -> None:
    payload = {
        "budget": {
            "gemini": {
                "daily_used": 10,
                "daily_hard_limit": 18,
            },
            "groq": {
                "available": False,
            },
        },
        "pairs": {
            "USD_JPY": {
                "gemini_status": {
                    "budget_used": 1,
                    "budget_limit": 99,
                },
                "groq_vote": {
                    "reasoning": "Groq未使用",
                },
            }
        },
    }

    snapshot = dashboard._derive_budget_snapshot(payload)

    assert snapshot["gemini"]["daily_used"] == 10
    assert snapshot["gemini"]["daily_hard_limit"] == 18
    assert snapshot["gemini"]["daily_remaining"] == 8
    assert snapshot["groq"]["available"] is False
    assert snapshot["groq"]["daily_remaining_label"] == "停止中"


def test_derive_budget_snapshot_uses_explicit_groq_numeric_budget() -> None:
    payload = {
        "budget": {
            "groq": {
                "available": True,
                "daily_used": 7,
                "daily_hard_limit": 20,
            }
        }
    }

    snapshot = dashboard._derive_budget_snapshot(payload)

    assert snapshot["groq"]["available"] is True
    assert snapshot["groq"]["daily_used"] == 7
    assert snapshot["groq"]["daily_hard_limit"] == 20
    assert snapshot["groq"]["daily_remaining"] == 13
    assert snapshot["groq"]["daily_remaining_label"] == "13/20"


def test_derive_budget_snapshot_groq_unavailable_hides_numeric_remaining() -> None:
    payload = {
        "budget": {
            "groq": {
                "available": False,
                "daily_used": 0,
                "daily_hard_limit": 9999,
            }
        }
    }

    snapshot = dashboard._derive_budget_snapshot(payload)

    assert snapshot["groq"]["available"] is False
    assert snapshot["groq"]["daily_hard_limit"] == 9999
    assert snapshot["groq"]["daily_remaining"] is None
    assert snapshot["groq"]["daily_remaining_label"] == "停止中"