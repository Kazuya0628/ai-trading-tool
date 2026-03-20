"""Multi-Timeframe Trading Strategy Engine.

Combines multiple analysis methods (AI Sennin's approach):
1. Technical pattern detection (algorithmic)
2. AI visual chart analysis (Gemini)
3. Multi-timeframe confirmation
4. Indicator confluence scoring

Generates final trade signals with entry/SL/TP levels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from src.strategies.ai_analyzer import AIChartAnalyzer
from src.strategies.pattern_detector import (
    PatternDetector,
    PatternSignal,
    PatternType,
    SignalDirection,
)


@dataclass
class TradeSignal:
    """Final trade signal combining all analysis methods."""
    epic: str
    pair_name: str
    direction: SignalDirection
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    pattern: PatternType
    timeframe: str
    timestamp: datetime = field(default_factory=datetime.now)
    reasoning: str = ""
    sources: list[str] = field(default_factory=list)
    mtf_confirmed: bool = False
    ai_confirmed: bool = False

    @property
    def is_actionable(self) -> bool:
        """Whether the signal meets minimum criteria for trading."""
        return (
            self.direction != SignalDirection.NONE
            and self.confidence >= 60
            and self.stop_loss > 0
            and self.take_profit > 0
        )

    @property
    def risk_reward_ratio(self) -> float:
        """Calculate risk:reward ratio."""
        risk = abs(self.entry_price - self.stop_loss)
        reward = abs(self.take_profit - self.entry_price)
        return reward / risk if risk > 0 else 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "epic": self.epic,
            "pair": self.pair_name,
            "direction": self.direction.value,
            "entry": self.entry_price,
            "sl": self.stop_loss,
            "tp": self.take_profit,
            "confidence": self.confidence,
            "pattern": self.pattern.value,
            "timeframe": self.timeframe,
            "timestamp": self.timestamp.isoformat(),
            "rr_ratio": round(self.risk_reward_ratio, 2),
            "mtf_confirmed": self.mtf_confirmed,
            "ai_confirmed": self.ai_confirmed,
            "reasoning": self.reasoning,
        }


class StrategyEngine:
    """Multi-strategy trading engine.

    Implements the full analysis pipeline:
    1. Fetch multi-timeframe data
    2. Calculate technical indicators
    3. Detect chart patterns (algorithmic)
    4. AI visual analysis (Gemini)
    5. Multi-timeframe confirmation
    6. Score and rank signals
    7. Generate final trade signals
    """

    def __init__(
        self,
        pattern_detector: PatternDetector,
        ai_analyzer: AIChartAnalyzer | None = None,
        config: dict | None = None,
    ) -> None:
        """Initialize Strategy Engine.

        Args:
            pattern_detector: Pattern detection module.
            ai_analyzer: AI visual analysis module.
            config: Strategy configuration.
        """
        self.pattern_detector = pattern_detector
        self.ai_analyzer = ai_analyzer
        self.config = config or {}

    def analyze(
        self,
        epic: str,
        pair_name: str,
        mtf_data: dict[str, pd.DataFrame],
        timeframe_config: dict[str, str],
    ) -> list[TradeSignal]:
        """Run full analysis pipeline on a currency pair.

        Args:
            epic: IG market epic.
            pair_name: Human-readable pair name.
            mtf_data: Multi-timeframe data {name: DataFrame}.
            timeframe_config: Timeframe configuration {name: resolution}.

        Returns:
            List of TradeSignal objects, sorted by confidence.
        """
        signals: list[TradeSignal] = []

        # 1. Get trend direction from higher timeframe
        trend_direction = "NEUTRAL"
        if "trend" in mtf_data:
            from src.data.indicators import TechnicalAnalyzer
            analyzer = TechnicalAnalyzer(self.config.get("indicators", {}))
            trend_df = analyzer.add_all_indicators(mtf_data["trend"])
            trend_direction = analyzer.get_trend_direction(trend_df)
            logger.info(f"{pair_name} trend: {trend_direction}")

        # 2. Detect patterns on primary timeframe
        primary_tf = timeframe_config.get("primary", "HOUR_4")
        if "primary" in mtf_data:
            primary_df = mtf_data["primary"]
            pattern_signals = self.pattern_detector.detect_all_patterns(primary_df)

            for ps in pattern_signals:
                if not ps.is_valid:
                    continue

                # Check trend alignment
                trend_aligned = self._check_trend_alignment(ps.direction, trend_direction)

                # Multi-timeframe confirmation
                mtf_confirmed = self._mtf_confirmation(
                    ps.direction, mtf_data, timeframe_config
                )

                # Build trade signal
                confidence = ps.confidence
                sources = ["pattern_detection"]

                if trend_aligned:
                    confidence += 10
                    sources.append("trend_aligned")
                else:
                    confidence -= 10

                if mtf_confirmed:
                    confidence += 15
                    sources.append("mtf_confirmed")

                confidence = max(0, min(100, confidence))

                ts = TradeSignal(
                    epic=epic,
                    pair_name=pair_name,
                    direction=ps.direction,
                    entry_price=ps.entry_price,
                    stop_loss=ps.stop_loss,
                    take_profit=ps.take_profit,
                    confidence=confidence,
                    pattern=ps.pattern,
                    timeframe=primary_tf,
                    reasoning=ps.reasoning,
                    sources=sources,
                    mtf_confirmed=mtf_confirmed,
                )
                signals.append(ts)

        # 3. AI Visual Analysis (if enabled)
        if self.ai_analyzer and "primary" in mtf_data:
            ai_signal = self._run_ai_analysis(
                mtf_data["primary"], pair_name, primary_tf
            )
            if ai_signal and ai_signal.is_valid:
                # Check if AI agrees with any pattern signal
                ai_agrees = any(
                    s.direction == ai_signal.direction for s in signals
                )
                if ai_agrees:
                    # Boost existing signals
                    for s in signals:
                        if s.direction == ai_signal.direction:
                            s.confidence = min(s.confidence + 10, 95)
                            s.ai_confirmed = True
                            s.sources.append("ai_visual_confirmed")
                else:
                    # Add AI signal as independent
                    confidence = ai_signal.confidence
                    if self._check_trend_alignment(ai_signal.direction, trend_direction):
                        confidence += 5

                    ts = TradeSignal(
                        epic=epic,
                        pair_name=pair_name,
                        direction=ai_signal.direction,
                        entry_price=ai_signal.entry_price,
                        stop_loss=ai_signal.stop_loss,
                        take_profit=ai_signal.take_profit,
                        confidence=confidence,
                        pattern=ai_signal.pattern,
                        timeframe=primary_tf,
                        reasoning=ai_signal.reasoning,
                        sources=["ai_visual_analysis"],
                        ai_confirmed=True,
                    )
                    signals.append(ts)

        # 4. Apply indicator confluence scoring
        if "primary" in mtf_data:
            self._apply_indicator_confluence(signals, mtf_data["primary"])

        # 5. Sort by confidence
        signals.sort(key=lambda s: s.confidence, reverse=True)

        # Log signals
        for s in signals:
            logger.info(
                f"Signal: {pair_name} {s.direction.value} "
                f"conf={s.confidence:.0f}% pattern={s.pattern.value} "
                f"RR={s.risk_reward_ratio:.1f} sources={s.sources}"
            )

        return signals

    def _check_trend_alignment(
        self, direction: SignalDirection, trend: str
    ) -> bool:
        """Check if signal direction aligns with higher-TF trend.

        Args:
            direction: Signal direction.
            trend: Trend direction string.

        Returns:
            True if aligned.
        """
        if trend == "NEUTRAL":
            return True
        if direction == SignalDirection.BUY and trend == "BULLISH":
            return True
        if direction == SignalDirection.SELL and trend == "BEARISH":
            return True
        return False

    def _mtf_confirmation(
        self,
        direction: SignalDirection,
        mtf_data: dict[str, pd.DataFrame],
        timeframe_config: dict[str, str],
    ) -> bool:
        """Check multi-timeframe confirmation.

        Signal is confirmed if secondary timeframe shows alignment.

        Args:
            direction: Signal direction to confirm.
            mtf_data: Multi-timeframe data.
            timeframe_config: Timeframe configuration.

        Returns:
            True if confirmed across timeframes.
        """
        confirmations = 0
        total_checks = 0

        for tf_name, df in mtf_data.items():
            if tf_name == "primary":
                continue
            if df.empty or len(df) < 20:
                continue

            total_checks += 1

            # Check short-term momentum
            recent_close = df["close"].iloc[-5:].values
            if direction == SignalDirection.BUY:
                if recent_close[-1] > recent_close[0]:
                    confirmations += 1
            elif direction == SignalDirection.SELL:
                if recent_close[-1] < recent_close[0]:
                    confirmations += 1

        return confirmations >= max(1, total_checks // 2) if total_checks > 0 else False

    def _run_ai_analysis(
        self, df: pd.DataFrame, pair_name: str, timeframe: str
    ) -> PatternSignal | None:
        """Run AI visual chart analysis.

        Args:
            df: OHLCV DataFrame.
            pair_name: Currency pair name.
            timeframe: Chart timeframe.

        Returns:
            PatternSignal from AI or None.
        """
        if not self.ai_analyzer:
            return None

        try:
            signal = self.ai_analyzer.analyze_chart(
                df=df,
                pair_name=pair_name,
                timeframe=timeframe,
            )
            if signal.is_valid:
                logger.info(
                    f"AI detected: {signal.pattern.value} "
                    f"{signal.direction.value} conf={signal.confidence:.0f}%"
                )
            return signal
        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            return None

    def _apply_indicator_confluence(
        self, signals: list[TradeSignal], df: pd.DataFrame
    ) -> None:
        """Adjust signal confidence based on indicator confluence.

        Args:
            signals: List of trade signals to adjust.
            df: DataFrame with indicators.
        """
        if df.empty:
            return

        latest = df.iloc[-1]

        for signal in signals:
            bonus = 0

            # RSI confirmation
            if "rsi" in df.columns:
                rsi = float(latest["rsi"])
                if signal.direction == SignalDirection.BUY and rsi < 40:
                    bonus += 5  # Oversold supports buy
                elif signal.direction == SignalDirection.SELL and rsi > 60:
                    bonus += 5  # Overbought supports sell

            # MACD confirmation
            if "macd_histogram" in df.columns:
                hist = float(latest["macd_histogram"])
                if signal.direction == SignalDirection.BUY and hist > 0:
                    bonus += 3
                elif signal.direction == SignalDirection.SELL and hist < 0:
                    bonus += 3

            # Bollinger Band confirmation
            if "bb_pct" in df.columns:
                bb_pct = float(latest["bb_pct"])
                if signal.direction == SignalDirection.BUY and bb_pct < 0.2:
                    bonus += 3  # Near lower band
                elif signal.direction == SignalDirection.SELL and bb_pct > 0.8:
                    bonus += 3  # Near upper band

            signal.confidence = min(signal.confidence + bonus, 95)
