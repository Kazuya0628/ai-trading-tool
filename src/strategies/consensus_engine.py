"""Multi-AI Consensus Engine for entry and close decisions.

Voting members:
  1. Algorithm  — pattern detector (BUY / SELL / NONE)
  2. Gemini     — chart visual + news/economic macro context (BUY / SELL / NONE)
  3. Groq       — regime + sentiment directional bias (BUY / SELL / NEUTRAL)

Standard 3-voter mode:
  - Algorithm must vote BUY or SELL (gatekeeper)
  - 2 of 3 agree on same direction
  - Algorithm must be in the majority
  - Average confidence of majority >= threshold

2-voter fallback mode:
  - When Gemini is unavailable (budget, error, skipped)
  - Algorithm and Groq must agree
  - Average confidence >= 62
  - Flagged as fallback=True

Close decision:
  - Phase 1: log only
  - Phase 2+: Both Gemini and Groq signal opposite with conf >= 80
"""

from __future__ import annotations

from dataclasses import dataclass, field

from loguru import logger

from src.domain.enums import ConsensusMode
from src.strategies.pattern_detector import SignalDirection


@dataclass
class ConsensusVote:
    """Single directional vote from one source."""

    source: str
    direction: SignalDirection
    confidence: float
    reasoning: str = ""


@dataclass
class ConsensusResult:
    """Aggregated result of multi-AI majority voting."""

    direction: SignalDirection
    confidence: float
    votes: list[ConsensusVote] = field(default_factory=list)
    consensus_reached: bool = False
    agree_count: int = 0
    total_votes: int = 0
    mode: ConsensusMode = ConsensusMode.THREE_VOTE
    reject_reason: str | None = None

    @property
    def is_fallback(self) -> bool:
        return self.mode == ConsensusMode.FALLBACK_TWO_VOTE

    @property
    def summary(self) -> str:
        parts = [
            f"{v.source}={v.direction.value}({v.confidence:.0f}%)"
            for v in self.votes
        ]
        mode_tag = "3V" if self.mode == ConsensusMode.THREE_VOTE else "2V-FB"
        return f"[{mode_tag} {self.agree_count}/{self.total_votes}] " + ", ".join(parts)


class ConsensusEngine:
    """Algorithm-gatekeeper consensus engine with 2-voter fallback.

    DESIGN-001 v1.3: 閾値はコード固定値ではなく設定値として扱う。
    初期値はconfigから読み込み、system_state（週次レビュー後）で上書き可能。
    """

    MAJORITY = 2

    def __init__(self, config: dict | None = None) -> None:
        """Initialize ConsensusEngine.

        Args:
            config: consensus設定 (trading_config.yaml の consensus セクション)。
                    未指定時は設計書デフォルト値を使用。
        """
        cfg = config or {}
        self.three_vote_threshold: float = float(
            cfg.get("three_vote_entry_threshold_initial",
                    cfg.get("three_vote_entry_threshold", 55.0))
        )
        self.fallback_threshold: float = float(
            cfg.get("fallback_entry_threshold_initial",
                    cfg.get("fallback_entry_threshold", 62.0))
        )
        self.close_min_conf: float = float(
            cfg.get("ai_close_min_confidence", 80.0)
        )
        self.threshold_min: float = float(cfg.get("threshold_min", 50.0))
        self.threshold_max: float = float(cfg.get("threshold_max", 65.0))
        self.threshold_step: float = float(cfg.get("threshold_adjust_step", 5.0))

    # --- 後方互換クラス変数エイリアス（既存コードが直参照している場合の保険） ---
    @property
    def CLOSE_MIN_CONF(self) -> float:
        return self.close_min_conf

    @property
    def FALLBACK_MIN_AVG_CONF(self) -> float:
        return self.fallback_threshold

    @property
    def THREE_VOTE_MIN_AVG_CONF(self) -> float:
        return self.three_vote_threshold

    def update_thresholds(
        self,
        three_vote: float | None = None,
        fallback: float | None = None,
    ) -> None:
        """週次レビュー等で閾値を更新する（min/max範囲内にクランプ）。

        Args:
            three_vote: 新しい3者投票閾値。
            fallback: 新しいフォールバック閾値。
        """
        if three_vote is not None:
            self.three_vote_threshold = max(
                self.threshold_min, min(self.threshold_max, three_vote)
            )
        if fallback is not None:
            self.fallback_threshold = max(
                self.threshold_min, min(self.threshold_max, fallback)
            )
        logger.info(
            f"[Consensus] Thresholds updated: 3V={self.three_vote_threshold:.0f} "
            f"FB={self.fallback_threshold:.0f}"
        )

    def decide_entry(
        self,
        algo_dir: SignalDirection,
        algo_conf: float,
        gemini_dir: SignalDirection,
        gemini_conf: float,
        groq_dir: SignalDirection,
        groq_conf: float,
        gemini_available: bool = True,
    ) -> ConsensusResult:
        """Apply consensus voting for new trade entry.

        Standard 3-voter mode:
        - Algorithm must be BUY or SELL (gatekeeper)
        - 2 of 3 must agree on same direction
        - Algorithm must be in the majority
        - Average confidence of majority >= threshold

        2-voter fallback (when Gemini unavailable):
        - Algorithm and Groq must agree
        - Average confidence >= 62

        Args:
            algo_dir/conf: Algorithm (pattern detector) vote.
            gemini_dir/conf: Gemini (chart + macro) vote.
            groq_dir/conf: Groq (regime + sentiment) vote.
            gemini_available: False triggers 2-voter fallback mode.

        Returns:
            ConsensusResult with direction, mode, and reject_reason.
        """
        # Gate: Algorithm must have a directional signal
        if algo_dir not in (SignalDirection.BUY, SignalDirection.SELL):
            votes = [
                ConsensusVote("algorithm", algo_dir, algo_conf),
                ConsensusVote("gemini", gemini_dir, gemini_conf),
                ConsensusVote("groq", groq_dir, groq_conf),
            ]
            return ConsensusResult(
                direction=SignalDirection.NONE,
                confidence=0,
                votes=votes,
                consensus_reached=False,
                agree_count=0,
                total_votes=len(votes),
                reject_reason="Algorithm has no directional signal",
            )

        # 2-voter fallback mode
        if not gemini_available:
            return self._fallback_two_vote(algo_dir, algo_conf, groq_dir, groq_conf)

        # Standard 3-voter mode
        return self._standard_three_vote(
            algo_dir, algo_conf, gemini_dir, gemini_conf, groq_dir, groq_conf
        )

    def decide_close(
        self,
        position_direction: SignalDirection,
        gemini_dir: SignalDirection,
        gemini_conf: float,
        groq_dir: SignalDirection,
        groq_conf: float,
        phase: int = 1,
        gemini_fresh: bool = True,
    ) -> ConsensusResult:
        """Determine if an existing position should be closed.

        Phase 1: Log only (always returns consensus_reached=False).
        Phase 2+: Requires BOTH AIs to signal opposite with conf >= 80,
                  and Gemini must be fresh (within 20 min).

        Args:
            position_direction: Direction of the open position.
            gemini_dir/conf: Gemini re-evaluation vote.
            groq_dir/conf: Groq directional vote.
            phase: Current operational phase (1, 2, or 3).
            gemini_fresh: Whether Gemini analysis is within 20 min.

        Returns:
            ConsensusResult; direction == opposite means "close".
        """
        opposite = (
            SignalDirection.SELL
            if position_direction == SignalDirection.BUY
            else SignalDirection.BUY
        )

        votes = [
            ConsensusVote("gemini", gemini_dir, gemini_conf),
            ConsensusVote("groq", groq_dir, groq_conf),
        ]

        close_votes = [v for v in votes if v.direction == opposite]

        # Phase 1: log only, never actually close
        if phase <= 1:
            avg_conf = sum(v.confidence for v in votes) / len(votes) if votes else 50.0
            would_close = (
                len(close_votes) >= 2
                and min(v.confidence for v in close_votes) >= self.CLOSE_MIN_CONF
            )
            if would_close:
                logger.info(
                    f"[Consensus Close] Phase 1 shadow: WOULD close "
                    f"({', '.join(f'{v.source}={v.confidence:.0f}%' for v in close_votes)})"
                )
            return ConsensusResult(
                direction=position_direction,
                confidence=avg_conf,
                votes=votes,
                consensus_reached=False,
                agree_count=len(close_votes),
                total_votes=len(votes),
                reject_reason="Phase 1: log only",
            )

        # Phase 2+: actual close decision
        if len(close_votes) >= 2:
            min_conf = min(v.confidence for v in close_votes)
            avg_conf = sum(v.confidence for v in close_votes) / len(close_votes)

            if min_conf >= self.CLOSE_MIN_CONF:
                if not gemini_fresh:
                    logger.info(
                        "[Consensus Close] Blocked — Gemini analysis stale (>20 min)"
                    )
                    return ConsensusResult(
                        direction=position_direction,
                        confidence=avg_conf,
                        votes=votes,
                        consensus_reached=False,
                        agree_count=len(close_votes),
                        total_votes=len(votes),
                        reject_reason="Gemini analysis stale",
                    )

                result = ConsensusResult(
                    direction=opposite,
                    confidence=avg_conf,
                    votes=votes,
                    consensus_reached=True,
                    agree_count=len(close_votes),
                    total_votes=len(votes),
                    mode=ConsensusMode.THREE_VOTE,
                )
                logger.info(f"[Consensus Close] AI_CLOSE: {result.summary}")
                return result

            logger.info(
                f"[Consensus Close] Blocked — min_conf={min_conf:.0f}% < {self.CLOSE_MIN_CONF:.0f}%"
            )

        # Hold
        avg_conf = sum(v.confidence for v in votes) / len(votes) if votes else 50.0
        return ConsensusResult(
            direction=position_direction,
            confidence=avg_conf,
            votes=votes,
            consensus_reached=False,
            agree_count=max(0, len(votes) - len(close_votes)),
            total_votes=len(votes),
        )

    # --------------------------------------------------
    # Internal: 3-voter standard mode
    # --------------------------------------------------

    def _standard_three_vote(
        self,
        algo_dir: SignalDirection,
        algo_conf: float,
        gemini_dir: SignalDirection,
        gemini_conf: float,
        groq_dir: SignalDirection,
        groq_conf: float,
    ) -> ConsensusResult:
        """Standard 3-voter consensus with Algorithm as gatekeeper."""
        votes = [
            ConsensusVote("algorithm", algo_dir, algo_conf),
            ConsensusVote("gemini", gemini_dir, gemini_conf),
            ConsensusVote("groq", groq_dir, groq_conf),
        ]

        # Count by direction
        for target_dir in (SignalDirection.BUY, SignalDirection.SELL):
            group = [v for v in votes if v.direction == target_dir]
            if len(group) < self.MAJORITY:
                continue

            # Algorithm must be in the majority
            algo_in_group = any(v.source == "algorithm" for v in group)
            if not algo_in_group:
                continue

            avg_conf = sum(v.confidence for v in group) / len(group)
            if avg_conf < self.THREE_VOTE_MIN_AVG_CONF:
                return ConsensusResult(
                    direction=SignalDirection.NONE,
                    confidence=avg_conf,
                    votes=votes,
                    consensus_reached=False,
                    agree_count=len(group),
                    total_votes=len(votes),
                    mode=ConsensusMode.THREE_VOTE,
                    reject_reason=f"Average confidence {avg_conf:.0f}% < {self.THREE_VOTE_MIN_AVG_CONF:.0f}%",
                )

            result = ConsensusResult(
                direction=target_dir,
                confidence=avg_conf,
                votes=votes,
                consensus_reached=True,
                agree_count=len(group),
                total_votes=len(votes),
                mode=ConsensusMode.THREE_VOTE,
            )
            logger.info(f"[Consensus Entry] 3V: {result.summary}")
            return result

        # No consensus
        buy = [v for v in votes if v.direction == SignalDirection.BUY]
        sell = [v for v in votes if v.direction == SignalDirection.SELL]
        avg_conf = sum(v.confidence for v in votes) / len(votes)
        return ConsensusResult(
            direction=SignalDirection.NONE,
            confidence=avg_conf,
            votes=votes,
            consensus_reached=False,
            agree_count=max(len(buy), len(sell)),
            total_votes=len(votes),
            mode=ConsensusMode.THREE_VOTE,
            reject_reason="No 2/3 majority with Algorithm in majority",
        )

    # --------------------------------------------------
    # Internal: 2-voter fallback mode
    # --------------------------------------------------

    def _fallback_two_vote(
        self,
        algo_dir: SignalDirection,
        algo_conf: float,
        groq_dir: SignalDirection,
        groq_conf: float,
    ) -> ConsensusResult:
        """2-voter fallback when Gemini is unavailable."""
        votes = [
            ConsensusVote("algorithm", algo_dir, algo_conf),
            ConsensusVote("groq", groq_dir, groq_conf),
        ]

        if algo_dir == groq_dir and algo_dir in (SignalDirection.BUY, SignalDirection.SELL):
            avg_conf = (algo_conf + groq_conf) / 2

            if avg_conf < self.FALLBACK_MIN_AVG_CONF:
                result = ConsensusResult(
                    direction=SignalDirection.NONE,
                    confidence=avg_conf,
                    votes=votes,
                    consensus_reached=False,
                    agree_count=2,
                    total_votes=2,
                    mode=ConsensusMode.FALLBACK_TWO_VOTE,
                    reject_reason=(
                        f"Fallback avg confidence {avg_conf:.0f}% < {self.FALLBACK_MIN_AVG_CONF:.0f}%"
                    ),
                )
                logger.info(f"[Consensus Entry] 2V-FB rejected: {result.summary}")
                return result

            result = ConsensusResult(
                direction=algo_dir,
                confidence=avg_conf,
                votes=votes,
                consensus_reached=True,
                agree_count=2,
                total_votes=2,
                mode=ConsensusMode.FALLBACK_TWO_VOTE,
            )
            logger.info(f"[Consensus Entry] 2V-FB: {result.summary}")
            return result

        # No agreement
        avg_conf = (algo_conf + groq_conf) / 2
        return ConsensusResult(
            direction=SignalDirection.NONE,
            confidence=avg_conf,
            votes=votes,
            consensus_reached=False,
            agree_count=1,
            total_votes=2,
            mode=ConsensusMode.FALLBACK_TWO_VOTE,
            reject_reason="Algorithm and Groq disagree",
        )
