"""Scheduler — Layer 0/1/2 job orchestration.

Responsibilities:
- Layer 0 (5min): Lightweight monitoring and dashboard update
- Layer 1 (4H close): Full analysis + entry decision
- Layer 2 (5min): Position monitoring and trailing stop updates
- Daily reset, weekly review, reconciliation scheduling
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from loguru import logger


class Scheduler:
    """Layer 0/1/2 scheduler with 4H close detection.

    Execution timing:
    - Layer 0: Every 5 min + 10s offset
    - Layer 1: 4H close (00,04,08,12,16,20 UTC) + 15s offset
    - Layer 2: Every 5 min + 40s offset
    """

    H4_SLOTS = (0, 4, 8, 12, 16, 20)

    def __init__(self) -> None:
        self._last_confirmed_h4_slot: str = ""
        self._last_layer0_time: float = 0
        self._last_layer2_time: float = 0
        self._last_daily_reset: str = ""
        self._last_weekly_review: str = ""

        self._layer0_interval: int = 300  # 5 min
        self._layer2_interval: int = 300  # 5 min

    @staticmethod
    def current_4h_slot() -> str:
        """Return the most recent 4H candle close slot as 'YYYY-MM-DD HH:00' (UTC)."""
        now = datetime.now(timezone.utc)
        slot_hour = (now.hour // 4) * 4
        return now.strftime(f"%Y-%m-%d {slot_hour:02d}:00")

    @staticmethod
    def next_4h_display() -> str:
        """Return the next 4H close time for display (JST)."""
        now = datetime.now(timezone.utc)
        next_slot_hour = ((now.hour // 4) + 1) * 4
        next_time = now.replace(
            hour=0, minute=0, second=0, microsecond=0
        ) + timedelta(hours=next_slot_hour)
        jst = next_time + timedelta(hours=9)
        return jst.strftime("%H:%M")

    def is_new_4h_candle(self) -> bool:
        """Check if a new 4H candle has closed since last check.

        Returns:
            True if a new 4H slot is detected.
        """
        current_slot = self.current_4h_slot()
        if current_slot != self._last_confirmed_h4_slot:
            self._last_confirmed_h4_slot = current_slot
            return True
        return False

    def should_run_layer0(self) -> bool:
        """Check if Layer 0 (lightweight monitoring) should run."""
        now = time.time()
        if now - self._last_layer0_time >= self._layer0_interval:
            self._last_layer0_time = now
            return True
        return False

    def should_run_layer2(self) -> bool:
        """Check if Layer 2 (position monitoring) should run."""
        now = time.time()
        if now - self._last_layer2_time >= self._layer2_interval:
            self._last_layer2_time = now
            return True
        return False

    def should_daily_reset(self) -> bool:
        """Check if daily reset (Gemini budget, etc.) should run."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self._last_daily_reset != today:
            self._last_daily_reset = today
            return True
        return False

    def should_weekly_review(self) -> bool:
        """Check if weekly review should run (Monday)."""
        now = datetime.now()
        week_key = now.strftime("%Y-W%W")
        if self._last_weekly_review != week_key and now.weekday() == 0:
            self._last_weekly_review = week_key
            return True
        return False

    def run_loop(
        self,
        on_layer0: Callable[[], None],
        on_layer1: Callable[[], None],
        on_layer2: Callable[[], None],
        on_daily_reset: Callable[[], None] | None = None,
        on_weekly_review: Callable[[], None] | None = None,
        on_reconciliation: Callable[[], None] | None = None,
        check_running: Callable[[], bool] = lambda: True,
        sleep_seconds: int = 10,
    ) -> None:
        """Main scheduler loop.

        Args:
            on_layer0: Callback for Layer 0 (monitoring).
            on_layer1: Callback for Layer 1 (4H entry analysis).
            on_layer2: Callback for Layer 2 (position management).
            on_daily_reset: Callback for daily reset.
            on_weekly_review: Callback for weekly review.
            on_reconciliation: Callback for reconciliation.
            check_running: Returns False to stop the loop.
            sleep_seconds: Sleep between loop iterations.
        """
        # Run reconciliation at startup
        if on_reconciliation:
            try:
                on_reconciliation()
            except Exception as e:
                logger.error(f"[Scheduler] Startup reconciliation failed: {e}")

        logger.info("[Scheduler] Starting main loop")

        while check_running():
            try:
                # Layer 1: 4H close — highest priority
                if self.is_new_4h_candle():
                    logger.info("[Scheduler] Layer 1: 4H candle closed — running full analysis")
                    on_layer1()

                # Layer 0: Lightweight monitoring
                if self.should_run_layer0():
                    logger.debug("[Scheduler] Layer 0: monitoring cycle")
                    on_layer0()

                # Layer 2: Position management
                if self.should_run_layer2():
                    logger.debug("[Scheduler] Layer 2: position monitoring")
                    on_layer2()

                # Daily reset
                if self.should_daily_reset() and on_daily_reset:
                    logger.info("[Scheduler] Daily reset")
                    on_daily_reset()

                # Weekly review
                if self.should_weekly_review() and on_weekly_review:
                    logger.info("[Scheduler] Weekly review")
                    on_weekly_review()

            except Exception as e:
                logger.error(f"[Scheduler] Error in loop iteration: {e}")

            time.sleep(sleep_seconds)

        logger.info("[Scheduler] Loop stopped")
