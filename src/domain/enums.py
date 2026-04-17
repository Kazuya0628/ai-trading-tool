"""Domain enumerations."""

from __future__ import annotations

from enum import Enum


class RunMode(Enum):
    """Bot execution mode."""

    PAPER = "PAPER"
    BACKTEST = "BACKTEST"
    ANALYSIS = "ANALYSIS"
    SAFE_STOP = "SAFE_STOP"
    ANALYSIS_ONLY = "ANALYSIS_ONLY"


class Phase(Enum):
    """Operational phase for progressive risk activation."""

    PHASE_1 = 1  # Fixed 0.5% risk, adaptive is shadow-only
    PHASE_2 = 2  # Consecutive loss + DD factors active
    PHASE_3 = 3  # All adaptive factors active (regime, performance)


class ConsensusMode(Enum):
    """Consensus voting mode."""

    THREE_VOTE = "THREE_VOTE"
    FALLBACK_TWO_VOTE = "FALLBACK_TWO_VOTE"


class VoteSource(Enum):
    """Source of a consensus vote."""

    ALGORITHM = "algorithm"
    GEMINI = "gemini"
    GROQ = "groq"


class GeminiSkipReason(Enum):
    """Reason for skipping Gemini analysis."""

    NOT_H4_CLOSE = "not_h4_close"
    NO_PATTERN = "no_pattern"
    SCORE_TOO_LOW = "score_too_low"
    BUDGET_EXHAUSTED = "budget_exhausted"
    NOT_SELECTED_BY_PRIORITY = "not_selected_by_priority"
    API_ERROR = "api_error"
