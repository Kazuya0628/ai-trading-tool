"""Chart pattern detection module.

Implements AI Sennin's pattern-based trading methodology:
- Double Bottom (W-shape) with 'right-side up' variant
- Double Top (M-shape)
- Inverse Head & Shoulders with 'right shoulder up'
- Head & Shoulders
- Channel Breakout
- Moving Average Crossover
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger


class PatternType(Enum):
    """Types of chart patterns to detect."""
    DOUBLE_BOTTOM = "double_bottom"
    DOUBLE_TOP = "double_top"
    INVERSE_HEAD_SHOULDERS = "inverse_head_shoulders"
    HEAD_SHOULDERS = "head_shoulders"
    CHANNEL_BREAKOUT = "channel_breakout"
    MA_CROSSOVER = "ma_crossover"
    NO_SIGNAL = "no_signal"


class SignalDirection(Enum):
    """Trade signal direction."""
    BUY = "BUY"
    SELL = "SELL"
    NONE = "NONE"


@dataclass
class PatternSignal:
    """Represents a detected pattern and its trading signal."""
    pattern: PatternType
    direction: SignalDirection
    confidence: float  # 0-100
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    neckline: float = 0.0
    support: float = 0.0
    resistance: float = 0.0
    reasoning: str = ""
    bar_index: int = -1
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return self.pattern != PatternType.NO_SIGNAL and self.confidence > 0

    @property
    def signal_code(self) -> int:
        """Binary signal: 1=trade, 0=no trade."""
        return 1 if self.is_valid else 0


class PatternDetector:
    """Detect chart patterns in OHLCV data.

    Implements the systematic pattern detection loop:
    1. Scan price data for structural patterns
    2. Validate with neckline breaks
    3. Generate signals with entry/SL/TP levels
    """

    def __init__(self, config: dict | None = None) -> None:
        """Initialize PatternDetector.

        Args:
            config: Pattern recognition configuration.
        """
        self.config = config or {}

    def detect_all_patterns(self, df: pd.DataFrame) -> list[PatternSignal]:
        """Scan for all enabled patterns in the data.

        Args:
            df: OHLCV DataFrame with indicators.

        Returns:
            List of detected PatternSignal objects.
        """
        if df.empty or len(df) < 30:
            return []

        signals = []
        enabled = self.config.get("enabled_patterns", [
            "double_bottom", "double_top", "inverse_head_shoulders",
            "head_shoulders", "channel_breakout", "ma_crossover",
        ])

        if "double_bottom" in enabled:
            signal = self.detect_double_bottom(df)
            if signal.is_valid:
                signals.append(signal)

        if "double_top" in enabled:
            signal = self.detect_double_top(df)
            if signal.is_valid:
                signals.append(signal)

        if "inverse_head_shoulders" in enabled:
            signal = self.detect_inverse_head_shoulders(df)
            if signal.is_valid:
                signals.append(signal)

        if "head_shoulders" in enabled:
            signal = self.detect_head_shoulders(df)
            if signal.is_valid:
                signals.append(signal)

        if "channel_breakout" in enabled:
            signal = self.detect_channel_breakout(df)
            if signal.is_valid:
                signals.append(signal)

        if "ma_crossover" in enabled:
            signal = self.detect_ma_crossover(df)
            if signal.is_valid:
                signals.append(signal)

        return signals

    def detect_double_bottom(self, df: pd.DataFrame) -> PatternSignal:
        """Detect Double Bottom (W-shape) pattern.

        AI Sennin's preferred variant: 'Right-side up' where
        the second trough is higher than the first.

        Args:
            df: OHLCV DataFrame.

        Returns:
            PatternSignal for detected pattern.
        """
        cfg = self.config.get("double_bottom", {})
        min_bars = cfg.get("min_pattern_bars", 20)
        max_bars = cfg.get("max_pattern_bars", 100)
        prefer_higher_low = cfg.get("prefer_higher_low", True)

        if len(df) < min_bars:
            return PatternSignal(PatternType.NO_SIGNAL, SignalDirection.NONE, 0)

        # Look at recent data window
        window = df.iloc[-max_bars:]
        close = window["close"].values
        low = window["low"].values
        high = window["high"].values

        # Find swing lows (potential bottoms)
        swing_lows = self._find_swing_points(low, order=5, mode="low")

        if len(swing_lows) < 2:
            return PatternSignal(PatternType.NO_SIGNAL, SignalDirection.NONE, 0)

        # Take last two swing lows
        trough1_idx, trough1_val = swing_lows[-2]
        trough2_idx, trough2_val = swing_lows[-1]

        # Distance check
        bar_distance = trough2_idx - trough1_idx
        if bar_distance < min_bars // 2 or bar_distance > max_bars:
            return PatternSignal(PatternType.NO_SIGNAL, SignalDirection.NONE, 0)

        # Find the neckline (highest point between troughs)
        between = high[trough1_idx:trough2_idx + 1]
        if len(between) == 0:
            return PatternSignal(PatternType.NO_SIGNAL, SignalDirection.NONE, 0)
        neckline_val = float(np.max(between))

        # Pattern depth check (at least 1% move)
        depth = (neckline_val - min(trough1_val, trough2_val)) / neckline_val
        if depth < 0.005:  # 0.5% minimum depth
            return PatternSignal(PatternType.NO_SIGNAL, SignalDirection.NONE, 0)

        # Troughs should be at similar levels (within 30% of depth)
        trough_diff = abs(trough1_val - trough2_val) / neckline_val
        if trough_diff > depth * 0.5:
            return PatternSignal(PatternType.NO_SIGNAL, SignalDirection.NONE, 0)

        # Calculate confidence
        confidence = 50.0

        # Bonus: 'right-side up' variant (second trough higher)
        if prefer_higher_low and trough2_val > trough1_val:
            confidence += 15.0

        # Bonus: neckline break confirmation
        current_close = float(close[-1])
        if current_close > neckline_val:
            confidence += 20.0
        elif current_close > neckline_val * 0.998:  # Within 0.2% of neckline
            confidence += 10.0

        # Bonus: volume increase (if available)
        if "volume" in window.columns:
            vol = window["volume"].values
            recent_vol = np.mean(vol[-5:])
            avg_vol = np.mean(vol[-20:])
            if recent_vol > avg_vol * 1.2:
                confidence += 5.0

        # RSI divergence bonus
        if "rsi" in window.columns:
            rsi = window["rsi"].values
            if len(rsi) > trough2_idx and len(rsi) > trough1_idx:
                if rsi[trough2_idx] > rsi[trough1_idx] and trough2_val <= trough1_val:
                    confidence += 10.0  # Bullish divergence

        confidence = min(confidence, 95.0)

        if confidence < 50:
            return PatternSignal(PatternType.NO_SIGNAL, SignalDirection.NONE, 0)

        # Calculate levels
        pattern_height = neckline_val - min(trough1_val, trough2_val)
        stop_loss = min(trough1_val, trough2_val) - pattern_height * 0.1
        take_profit = neckline_val + pattern_height  # Measured move

        return PatternSignal(
            pattern=PatternType.DOUBLE_BOTTOM,
            direction=SignalDirection.BUY,
            confidence=confidence,
            entry_price=current_close,
            stop_loss=stop_loss,
            take_profit=take_profit,
            neckline=neckline_val,
            support=min(trough1_val, trough2_val),
            resistance=neckline_val,
            reasoning=(
                f"Double Bottom detected. "
                f"Trough1={trough1_val:.5f}, Trough2={trough2_val:.5f}, "
                f"Neckline={neckline_val:.5f}. "
                f"{'Higher low (preferred variant)' if trough2_val > trough1_val else 'Equal lows'}"
            ),
            bar_index=len(df) - 1,
        )

    def detect_double_top(self, df: pd.DataFrame) -> PatternSignal:
        """Detect Double Top (M-shape) pattern.

        Args:
            df: OHLCV DataFrame.

        Returns:
            PatternSignal for detected pattern.
        """
        cfg = self.config.get("double_top", {})
        min_bars = cfg.get("min_pattern_bars", 20)
        max_bars = cfg.get("max_pattern_bars", 100)

        if len(df) < min_bars:
            return PatternSignal(PatternType.NO_SIGNAL, SignalDirection.NONE, 0)

        window = df.iloc[-max_bars:]
        close = window["close"].values
        high = window["high"].values
        low = window["low"].values

        swing_highs = self._find_swing_points(high, order=5, mode="high")

        if len(swing_highs) < 2:
            return PatternSignal(PatternType.NO_SIGNAL, SignalDirection.NONE, 0)

        peak1_idx, peak1_val = swing_highs[-2]
        peak2_idx, peak2_val = swing_highs[-1]

        bar_distance = peak2_idx - peak1_idx
        if bar_distance < min_bars // 2 or bar_distance > max_bars:
            return PatternSignal(PatternType.NO_SIGNAL, SignalDirection.NONE, 0)

        between = low[peak1_idx:peak2_idx + 1]
        if len(between) == 0:
            return PatternSignal(PatternType.NO_SIGNAL, SignalDirection.NONE, 0)
        neckline_val = float(np.min(between))

        depth = (max(peak1_val, peak2_val) - neckline_val) / max(peak1_val, peak2_val)
        if depth < 0.005:
            return PatternSignal(PatternType.NO_SIGNAL, SignalDirection.NONE, 0)

        peak_diff = abs(peak1_val - peak2_val) / max(peak1_val, peak2_val)
        if peak_diff > depth * 0.5:
            return PatternSignal(PatternType.NO_SIGNAL, SignalDirection.NONE, 0)

        confidence = 50.0
        current_close = float(close[-1])

        # Lower second peak = stronger
        if peak2_val < peak1_val:
            confidence += 15.0

        if current_close < neckline_val:
            confidence += 20.0
        elif current_close < neckline_val * 1.002:
            confidence += 10.0

        confidence = min(confidence, 95.0)

        if confidence < 50:
            return PatternSignal(PatternType.NO_SIGNAL, SignalDirection.NONE, 0)

        pattern_height = max(peak1_val, peak2_val) - neckline_val
        stop_loss = max(peak1_val, peak2_val) + pattern_height * 0.1
        take_profit = neckline_val - pattern_height

        return PatternSignal(
            pattern=PatternType.DOUBLE_TOP,
            direction=SignalDirection.SELL,
            confidence=confidence,
            entry_price=current_close,
            stop_loss=stop_loss,
            take_profit=take_profit,
            neckline=neckline_val,
            support=neckline_val,
            resistance=max(peak1_val, peak2_val),
            reasoning=(
                f"Double Top detected. "
                f"Peak1={peak1_val:.5f}, Peak2={peak2_val:.5f}, "
                f"Neckline={neckline_val:.5f}"
            ),
            bar_index=len(df) - 1,
        )

    def detect_inverse_head_shoulders(self, df: pd.DataFrame) -> PatternSignal:
        """Detect Inverse Head & Shoulders pattern.

        AI Sennin prefers the 'right shoulder up' variant.

        Args:
            df: OHLCV DataFrame.

        Returns:
            PatternSignal for detected pattern.
        """
        cfg = self.config.get("inverse_head_shoulders", {})
        min_bars = cfg.get("min_pattern_bars", 30)
        max_bars = cfg.get("max_pattern_bars", 150)

        if len(df) < min_bars:
            return PatternSignal(PatternType.NO_SIGNAL, SignalDirection.NONE, 0)

        window = df.iloc[-max_bars:]
        low = window["low"].values
        high = window["high"].values
        close = window["close"].values

        swing_lows = self._find_swing_points(low, order=7, mode="low")

        if len(swing_lows) < 3:
            return PatternSignal(PatternType.NO_SIGNAL, SignalDirection.NONE, 0)

        # Last three swing lows: left shoulder, head, right shoulder
        ls_idx, ls_val = swing_lows[-3]
        head_idx, head_val = swing_lows[-2]
        rs_idx, rs_val = swing_lows[-1]

        # Head must be lower than both shoulders
        if head_val >= ls_val or head_val >= rs_val:
            return PatternSignal(PatternType.NO_SIGNAL, SignalDirection.NONE, 0)

        # Shoulders should be at similar levels
        shoulder_diff = abs(ls_val - rs_val) / max(ls_val, rs_val)
        if shoulder_diff > 0.03:  # 3% tolerance
            return PatternSignal(PatternType.NO_SIGNAL, SignalDirection.NONE, 0)

        # Calculate neckline from peaks between shoulders and head
        between1 = high[ls_idx:head_idx + 1]
        between2 = high[head_idx:rs_idx + 1]
        if len(between1) == 0 or len(between2) == 0:
            return PatternSignal(PatternType.NO_SIGNAL, SignalDirection.NONE, 0)

        neckline_left = float(np.max(between1))
        neckline_right = float(np.max(between2))
        neckline = (neckline_left + neckline_right) / 2

        confidence = 55.0
        current_close = float(close[-1])

        # 'Right shoulder up' variant (preferred by AI Sennin)
        if rs_val > ls_val:
            confidence += 15.0

        # Neckline break
        if current_close > neckline:
            confidence += 20.0

        confidence = min(confidence, 95.0)

        if confidence < 50:
            return PatternSignal(PatternType.NO_SIGNAL, SignalDirection.NONE, 0)

        pattern_height = neckline - head_val
        stop_loss = head_val - pattern_height * 0.1
        take_profit = neckline + pattern_height

        return PatternSignal(
            pattern=PatternType.INVERSE_HEAD_SHOULDERS,
            direction=SignalDirection.BUY,
            confidence=confidence,
            entry_price=current_close,
            stop_loss=stop_loss,
            take_profit=take_profit,
            neckline=neckline,
            support=head_val,
            resistance=neckline,
            reasoning=(
                f"Inverse H&S detected. "
                f"LS={ls_val:.5f}, Head={head_val:.5f}, RS={rs_val:.5f}, "
                f"Neckline={neckline:.5f}. "
                f"{'Right shoulder up (preferred)' if rs_val > ls_val else 'Classic form'}"
            ),
            bar_index=len(df) - 1,
        )

    def detect_head_shoulders(self, df: pd.DataFrame) -> PatternSignal:
        """Detect Head & Shoulders (bearish) pattern.

        Args:
            df: OHLCV DataFrame.

        Returns:
            PatternSignal for detected pattern.
        """
        if len(df) < 30:
            return PatternSignal(PatternType.NO_SIGNAL, SignalDirection.NONE, 0)

        max_bars = self.config.get("head_shoulders", {}).get("max_pattern_bars", 150)
        window = df.iloc[-max_bars:]
        high = window["high"].values
        low = window["low"].values
        close = window["close"].values

        swing_highs = self._find_swing_points(high, order=7, mode="high")

        if len(swing_highs) < 3:
            return PatternSignal(PatternType.NO_SIGNAL, SignalDirection.NONE, 0)

        ls_idx, ls_val = swing_highs[-3]
        head_idx, head_val = swing_highs[-2]
        rs_idx, rs_val = swing_highs[-1]

        if head_val <= ls_val or head_val <= rs_val:
            return PatternSignal(PatternType.NO_SIGNAL, SignalDirection.NONE, 0)

        shoulder_diff = abs(ls_val - rs_val) / max(ls_val, rs_val)
        if shoulder_diff > 0.03:
            return PatternSignal(PatternType.NO_SIGNAL, SignalDirection.NONE, 0)

        between1 = low[ls_idx:head_idx + 1]
        between2 = low[head_idx:rs_idx + 1]
        if len(between1) == 0 or len(between2) == 0:
            return PatternSignal(PatternType.NO_SIGNAL, SignalDirection.NONE, 0)

        neckline = (float(np.min(between1)) + float(np.min(between2))) / 2

        confidence = 55.0
        current_close = float(close[-1])

        if rs_val < ls_val:
            confidence += 10.0
        if current_close < neckline:
            confidence += 20.0

        confidence = min(confidence, 95.0)

        if confidence < 50:
            return PatternSignal(PatternType.NO_SIGNAL, SignalDirection.NONE, 0)

        pattern_height = head_val - neckline
        stop_loss = head_val + pattern_height * 0.1
        take_profit = neckline - pattern_height

        return PatternSignal(
            pattern=PatternType.HEAD_SHOULDERS,
            direction=SignalDirection.SELL,
            confidence=confidence,
            entry_price=current_close,
            stop_loss=stop_loss,
            take_profit=take_profit,
            neckline=neckline,
            support=neckline,
            resistance=head_val,
            reasoning=f"H&S: LS={ls_val:.5f}, Head={head_val:.5f}, RS={rs_val:.5f}",
            bar_index=len(df) - 1,
        )

    def detect_channel_breakout(self, df: pd.DataFrame) -> PatternSignal:
        """Detect price channel breakout.

        Args:
            df: OHLCV DataFrame.

        Returns:
            PatternSignal for detected pattern.
        """
        cfg = self.config.get("channel_breakout", {})
        lookback = cfg.get("lookback_period", 20)
        atr_mult = cfg.get("breakout_threshold_atr_multiple", 1.5)

        if len(df) < lookback + 5:
            return PatternSignal(PatternType.NO_SIGNAL, SignalDirection.NONE, 0)

        # Channel boundaries from lookback period (excluding latest 2 bars)
        channel_data = df.iloc[-(lookback + 2):-2]
        channel_high = float(channel_data["high"].max())
        channel_low = float(channel_data["low"].min())

        current_close = float(df["close"].iloc[-1])
        prev_close = float(df["close"].iloc[-2])  # previous bar's close
        atr = float(df["atr"].iloc[-1]) if "atr" in df.columns else 0

        # Bullish breakout — require FRESH breakout (prev bar was still inside channel)
        if current_close > channel_high and prev_close <= channel_high:
            breakout_size = current_close - channel_high
            if atr > 0 and breakout_size >= atr * 0.5:
                confidence = min(50 + (breakout_size / atr) * 20, 90)
                channel_width = channel_high - channel_low
                return PatternSignal(
                    pattern=PatternType.CHANNEL_BREAKOUT,
                    direction=SignalDirection.BUY,
                    confidence=confidence,
                    entry_price=current_close,
                    stop_loss=channel_low,
                    take_profit=current_close + channel_width,
                    support=channel_low,
                    resistance=channel_high,
                    reasoning=f"Fresh bullish breakout above {channel_high:.5f}",
                    bar_index=len(df) - 1,
                )

        # Bearish breakout — require FRESH breakout (prev bar was still inside channel)
        if current_close < channel_low and prev_close >= channel_low:
            breakout_size = channel_low - current_close
            if atr > 0 and breakout_size >= atr * 0.5:
                confidence = min(50 + (breakout_size / atr) * 20, 90)
                channel_width = channel_high - channel_low
                return PatternSignal(
                    pattern=PatternType.CHANNEL_BREAKOUT,
                    direction=SignalDirection.SELL,
                    confidence=confidence,
                    entry_price=current_close,
                    stop_loss=channel_high,
                    take_profit=current_close - channel_width,
                    support=channel_low,
                    resistance=channel_high,
                    reasoning=f"Fresh bearish breakout below {channel_low:.5f}",
                    bar_index=len(df) - 1,
                )

        return PatternSignal(PatternType.NO_SIGNAL, SignalDirection.NONE, 0)

    def detect_ma_crossover(self, df: pd.DataFrame) -> PatternSignal:
        """Detect Moving Average crossover signals.

        Args:
            df: OHLCV DataFrame with MA indicators.

        Returns:
            PatternSignal for detected crossover.
        """
        cfg = self.config.get("ma_crossover", {})
        fast_period = cfg.get("fast_period", 20)
        slow_period = cfg.get("slow_period", 50)

        fast_col = f"sma_{fast_period}"
        slow_col = f"sma_{slow_period}"

        # Fall back to EMA if SMA not available
        if fast_col not in df.columns:
            fast_col = f"ema_{fast_period}" if f"ema_{fast_period}" in df.columns else None
        if slow_col not in df.columns:
            slow_col = f"ema_{slow_period}" if f"ema_{slow_period}" in df.columns else None

        if not fast_col or not slow_col or fast_col not in df.columns or slow_col not in df.columns:
            return PatternSignal(PatternType.NO_SIGNAL, SignalDirection.NONE, 0)

        if len(df) < 3:
            return PatternSignal(PatternType.NO_SIGNAL, SignalDirection.NONE, 0)

        # Check for crossover in last 2 bars
        fast_prev = float(df[fast_col].iloc[-2])
        fast_curr = float(df[fast_col].iloc[-1])
        slow_prev = float(df[slow_col].iloc[-2])
        slow_curr = float(df[slow_col].iloc[-1])

        current_close = float(df["close"].iloc[-1])
        atr = float(df["atr"].iloc[-1]) if "atr" in df.columns else 0

        # Golden cross (bullish)
        if fast_prev <= slow_prev and fast_curr > slow_curr:
            confidence = 60.0
            if "adx" in df.columns and float(df["adx"].iloc[-1]) > 25:
                confidence += 10.0

            stop_loss = current_close - (atr * 2 if atr else current_close * 0.01)
            take_profit = current_close + (atr * 4 if atr else current_close * 0.02)

            return PatternSignal(
                pattern=PatternType.MA_CROSSOVER,
                direction=SignalDirection.BUY,
                confidence=confidence,
                entry_price=current_close,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reasoning=f"Golden cross: {fast_col} crossed above {slow_col}",
                bar_index=len(df) - 1,
            )

        # Death cross (bearish)
        if fast_prev >= slow_prev and fast_curr < slow_curr:
            confidence = 60.0
            if "adx" in df.columns and float(df["adx"].iloc[-1]) > 25:
                confidence += 10.0

            stop_loss = current_close + (atr * 2 if atr else current_close * 0.01)
            take_profit = current_close - (atr * 4 if atr else current_close * 0.02)

            return PatternSignal(
                pattern=PatternType.MA_CROSSOVER,
                direction=SignalDirection.SELL,
                confidence=confidence,
                entry_price=current_close,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reasoning=f"Death cross: {fast_col} crossed below {slow_col}",
                bar_index=len(df) - 1,
            )

        return PatternSignal(PatternType.NO_SIGNAL, SignalDirection.NONE, 0)

    # --------------------------------------------------
    # Helper Methods
    # --------------------------------------------------

    @staticmethod
    def _find_swing_points(
        data: np.ndarray,
        order: int = 5,
        mode: str = "low",
    ) -> list[tuple[int, float]]:
        """Find swing high/low points in price data.

        Args:
            data: Price array (highs or lows).
            order: Number of bars on each side for comparison.
            mode: 'low' for swing lows, 'high' for swing highs.

        Returns:
            List of (index, value) tuples for swing points.
        """
        points = []
        for i in range(order, len(data) - order):
            if mode == "low":
                if all(data[i] <= data[i - j] for j in range(1, order + 1)) and \
                   all(data[i] <= data[i + j] for j in range(1, order + 1)):
                    points.append((i, float(data[i])))
            else:  # high
                if all(data[i] >= data[i - j] for j in range(1, order + 1)) and \
                   all(data[i] >= data[i + j] for j in range(1, order + 1)):
                    points.append((i, float(data[i])))
        return points
