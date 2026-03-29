"""Runtime State — centralized execution state for TradingBot.

Holds all mutable state that was previously scattered across TradingBot attributes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.domain.enums import Phase, RunMode


@dataclass
class RuntimeState:
    """Centralized execution state."""

    # Execution state
    running: bool = False
    run_mode: RunMode = RunMode.PAPER
    phase: Phase = Phase.PHASE_1

    # 4H management
    last_confirmed_h4_slot: str = ""
    next_4h_time: str = ""

    # Dashboard
    cycle_log: dict[str, dict[str, Any]] = field(default_factory=dict)
    last_cycle_started_at: str = ""
    last_cycle_finished_at: str = ""

    # AI caches
    gemini_cache: dict[str, dict[str, Any]] = field(default_factory=dict)
    groq_vote_cache: dict[str, dict[str, Any] | None] = field(default_factory=dict)

    # Safety
    safe_stop_reason: str | None = None

    # Tracking
    last_summary_date: str = ""
    last_weekly_review_date: str = ""
    dd_warned_levels: set[float] = field(default_factory=set)
    prev_regime: str = ""
    position_eval_counter: int = 0

    def enter_safe_stop(self, reason: str) -> None:
        """Transition to SAFE_STOP mode.

        Args:
            reason: Reason for entering safe stop.
        """
        self.run_mode = RunMode.SAFE_STOP
        self.safe_stop_reason = reason

    def enter_analysis_only(self, reason: str) -> None:
        """Transition to ANALYSIS_ONLY mode.

        Args:
            reason: Reason for entering analysis-only mode.
        """
        self.run_mode = RunMode.ANALYSIS_ONLY
        self.safe_stop_reason = reason

    def recover_to_paper(self) -> None:
        """Recover from ANALYSIS_ONLY back to PAPER mode."""
        if self.run_mode == RunMode.ANALYSIS_ONLY:
            self.run_mode = RunMode.PAPER
            self.safe_stop_reason = None

    @property
    def can_enter_trades(self) -> bool:
        """Whether the current mode allows new trade entries."""
        return self.run_mode == RunMode.PAPER and self.running
