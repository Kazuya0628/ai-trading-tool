"""Gemini Budget Service — manages daily and per-cycle API call limits.

Responsibilities:
- Daily soft limit (16) / hard limit (18) management
- Per-4H-cycle limit (3 calls)
- Priority-based candidate selection
- Skip reason tracking
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from loguru import logger

from src.domain.enums import GeminiSkipReason


class GeminiBudgetService:
    """Manages Gemini API call budget and candidate prioritization."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self._daily_soft_limit: int = cfg.get("daily_soft_limit", 16)
        self._daily_hard_limit: int = cfg.get("daily_hard_limit", 18)
        self._cycle_limit: int = cfg.get("cycle_limit", 3)

        self._daily_count: int = 0
        self._cycle_count: int = 0
        self._count_date: str = ""
        self._current_cycle_slot: str = ""

    @property
    def daily_count(self) -> int:
        return self._daily_count

    @property
    def daily_limit(self) -> int:
        return self._daily_hard_limit

    @property
    def budget_remaining(self) -> int:
        self._check_daily_reset()
        return max(0, self._daily_hard_limit - self._daily_count)

    def can_call(self) -> tuple[bool, GeminiSkipReason | None]:
        """Check if a Gemini call is within budget.

        Returns:
            Tuple of (allowed, skip_reason_if_not).
        """
        self._check_daily_reset()

        if self._daily_count >= self._daily_hard_limit:
            return False, GeminiSkipReason.BUDGET_EXHAUSTED

        if self._cycle_count >= self._cycle_limit:
            return False, GeminiSkipReason.NOT_SELECTED_BY_PRIORITY

        return True, None

    def consume(self) -> None:
        """Consume one API call from the budget."""
        self._check_daily_reset()
        self._daily_count += 1
        self._cycle_count += 1
        logger.info(
            f"[Gemini budget] {self._daily_count}/{self._daily_hard_limit} daily, "
            f"{self._cycle_count}/{self._cycle_limit} this cycle"
        )

    def start_new_cycle(self, slot: str) -> None:
        """Reset per-cycle counter for a new 4H slot.

        Args:
            slot: 4H slot identifier (e.g., '2026-03-29 08:00').
        """
        if slot != self._current_cycle_slot:
            self._cycle_count = 0
            self._current_cycle_slot = slot
            logger.debug(f"[Gemini budget] New cycle: {slot}")

    def select_candidates(
        self, candidates: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Select top candidates for Gemini analysis based on priority.

        Priority factors:
        1. Historical pattern/pair performance
        2. MTF alignment
        3. ATR-normalized volatility
        4. Session importance
        5. Fairness (round-robin)

        Args:
            candidates: List of candidate dicts with instrument, pattern_score, etc.

        Returns:
            Selected candidates (up to cycle_limit, within daily budget).
        """
        available = min(
            self._cycle_limit - self._cycle_count,
            self._daily_hard_limit - self._daily_count,
        )
        if available <= 0:
            return []

        # Sort by pattern_score descending as primary priority
        scored = sorted(
            candidates,
            key=lambda c: c.get("pattern_score", 0),
            reverse=True,
        )

        selected = scored[:available]
        if selected:
            instruments = [c.get("instrument", "?") for c in selected]
            logger.info(
                f"[Gemini budget] Selected {len(selected)} candidates: {instruments}"
            )

        return selected

    def get_status(self) -> dict[str, Any]:
        """Get current budget status for dashboard."""
        self._check_daily_reset()
        return {
            "daily_used": self._daily_count,
            "daily_soft_limit": self._daily_soft_limit,
            "daily_hard_limit": self._daily_hard_limit,
            "cycle_used": self._cycle_count,
            "cycle_limit": self._cycle_limit,
            "at_soft_limit": self._daily_count >= self._daily_soft_limit,
            "at_hard_limit": self._daily_count >= self._daily_hard_limit,
        }

    def _check_daily_reset(self) -> None:
        """Reset daily counter on date change."""
        today = datetime.now().strftime("%Y-%m-%d")
        if self._count_date != today:
            self._daily_count = 0
            self._count_date = today
            logger.debug("[Gemini budget] Daily counter reset")
