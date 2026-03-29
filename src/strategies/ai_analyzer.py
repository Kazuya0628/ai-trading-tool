"""AI Visual Chart Analysis using Google Gemini.

Implements AI Sennin's approach:
1. Generate chart image
2. Send to AI (Gemini) for pattern recognition
3. Get binary signal (1=trade, 0=no signal)
4. Execute based on AI classification
"""

from __future__ import annotations

import hashlib
import io
import json
import tempfile
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
from loguru import logger
from PIL import Image

from src.strategies.pattern_detector import PatternSignal, PatternType, SignalDirection


class AIChartAnalyzer:
    """Analyze charts using Google Gemini visual AI.

    Systematic Loop (AI Sennin's methodology):
    1. Generate Chart Image
    2. AI Analysis (Gemini)
    3. Result Code Return (1/0)
    4. Execution Decision
    """

    def __init__(self, api_key: str, config: dict | None = None) -> None:
        """Initialize AI Chart Analyzer.

        Args:
            api_key: Google Gemini API key.
            config: AI analysis configuration.
        """
        self.api_key = api_key
        self.config = config or {}
        self._model = None

        # In-memory cache: key -> {"result": ..., "expires_at": float}
        self._cache: dict[str, dict] = {}
        self._cache_ttl: int = self.config.get("cache_ttl_seconds", 3600)  # 1h default

        if api_key:
            self._init_model()

    def _init_model(self) -> None:
        """Initialize Gemini model."""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            model_name = self.config.get("model", "gemini-2.5-flash")
            self._model = genai.GenerativeModel(model_name)
            logger.info(f"Gemini model initialized: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            self._model = None

    def generate_chart_image(
        self,
        df: pd.DataFrame,
        pair_name: str = "",
        timeframe: str = "",
        include_indicators: bool = True,
    ) -> bytes | None:
        """Generate a candlestick chart image from OHLCV data.

        Args:
            df: OHLCV DataFrame.
            pair_name: Currency pair name for title.
            timeframe: Timeframe for title.
            include_indicators: Whether to add indicator overlays.

        Returns:
            PNG image bytes or None on failure.
        """
        try:
            if df.empty or len(df) < 10:
                return None

            # Prepare data for mplfinance
            chart_df = df.copy()
            chart_df.index = pd.DatetimeIndex(chart_df.index)

            # Rename columns to match mplfinance expectations
            chart_df = chart_df.rename(columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            })

            # Build additional plots
            addplots = []

            if include_indicators:
                # Moving averages
                for col in ["sma_20", "sma_50", "sma_200"]:
                    if col in df.columns:
                        chart_df[col] = df[col].values
                        color = {"sma_20": "blue", "sma_50": "orange", "sma_200": "red"}
                        addplots.append(
                            mpf.make_addplot(
                                chart_df[col],
                                color=color.get(col, "gray"),
                                width=1.0,
                            )
                        )

                # Bollinger Bands
                for col in ["bb_upper", "bb_lower"]:
                    if col in df.columns:
                        chart_df[col] = df[col].values
                        addplots.append(
                            mpf.make_addplot(
                                chart_df[col],
                                color="purple",
                                linestyle="--",
                                width=0.5,
                            )
                        )

                # --- Sub-plot indicators (RSI, ADX, MACD) ---
                # RSI panel (panel=2)
                if "rsi" in df.columns:
                    rsi_data = df["rsi"].copy()
                    chart_df["rsi"] = rsi_data.values
                    addplots.append(
                        mpf.make_addplot(
                            chart_df["rsi"], panel=2, color="cyan",
                            ylabel="RSI", width=0.8,
                        )
                    )
                    # Overbought / Oversold lines
                    chart_df["rsi_70"] = 70.0
                    chart_df["rsi_30"] = 30.0
                    addplots.append(
                        mpf.make_addplot(
                            chart_df["rsi_70"], panel=2, color="red",
                            linestyle="--", width=0.4,
                        )
                    )
                    addplots.append(
                        mpf.make_addplot(
                            chart_df["rsi_30"], panel=2, color="green",
                            linestyle="--", width=0.4,
                        )
                    )

                # ADX panel (panel=3)
                if "adx" in df.columns:
                    chart_df["adx"] = df["adx"].values
                    addplots.append(
                        mpf.make_addplot(
                            chart_df["adx"], panel=3, color="yellow",
                            ylabel="ADX", width=0.8,
                        )
                    )
                    chart_df["adx_25"] = 25.0
                    addplots.append(
                        mpf.make_addplot(
                            chart_df["adx_25"], panel=3, color="gray",
                            linestyle="--", width=0.4,
                        )
                    )

                # MACD panel (panel=4)
                if "macd" in df.columns and "macd_signal" in df.columns:
                    chart_df["macd"] = df["macd"].values
                    chart_df["macd_signal"] = df["macd_signal"].values
                    addplots.append(
                        mpf.make_addplot(
                            chart_df["macd"], panel=4, color="lime",
                            ylabel="MACD", width=0.7,
                        )
                    )
                    addplots.append(
                        mpf.make_addplot(
                            chart_df["macd_signal"], panel=4, color="red",
                            width=0.7,
                        )
                    )
                    if "macd_hist" in df.columns:
                        chart_df["macd_hist"] = df["macd_hist"].values
                        colors = [
                            "#26a69a" if v >= 0 else "#ef5350"
                            for v in chart_df["macd_hist"]
                        ]
                        addplots.append(
                            mpf.make_addplot(
                                chart_df["macd_hist"], panel=4,
                                type="bar", color=colors, width=0.5,
                            )
                        )

            # Chart settings
            width = self.config.get("chart_width", 1200)
            height = self.config.get("chart_height", 800)
            style = self.config.get("chart_style", "charles")

            kwargs = {
                "type": "candle",
                "style": style,
                "title": f"{pair_name} {timeframe}" if pair_name else "",
                "figsize": (width / 100, height / 100),
                "volume": bool("Volume" in chart_df.columns and chart_df["Volume"].sum() > 0),
                "returnfig": True,
            }

            if addplots:
                kwargs["addplot"] = addplots

            fig, axes = mpf.plot(chart_df[["Open", "High", "Low", "Close", "Volume"]], **kwargs)

            # Save to bytes
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)

            return buf.getvalue()

        except Exception as e:
            logger.error(f"Chart generation failed: {e}")
            return None

    def analyze_chart(
        self,
        image_bytes: bytes | None = None,
        df: pd.DataFrame | None = None,
        pair_name: str = "",
        timeframe: str = "",
        context_data: dict | None = None,
    ) -> PatternSignal:
        """Analyze chart image using Gemini AI.

        Args:
            image_bytes: Pre-generated chart image bytes.
            df: OHLCV DataFrame (will generate chart if image_bytes is None).
            pair_name: Currency pair name.
            timeframe: Chart timeframe.

        Returns:
            PatternSignal from AI analysis.
        """
        if not self._model:
            logger.warning("Gemini not initialized, skipping AI analysis")
            return PatternSignal(PatternType.NO_SIGNAL, SignalDirection.NONE, 0)

        # Generate chart image if not provided
        if image_bytes is None:
            if df is not None:
                image_bytes = self.generate_chart_image(df, pair_name, timeframe)
            if image_bytes is None:
                return PatternSignal(PatternType.NO_SIGNAL, SignalDirection.NONE, 0)

        # Cache check: key = hash(pair_name + timeframe + last bar timestamp)
        cache_key = self._make_cache_key("chart", pair_name, timeframe, df)
        cached = self._get_cache(cache_key)
        if cached is not None:
            logger.info(f"[Gemini] Cache hit: {pair_name} {timeframe} chart analysis")
            return cached

        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_bytes))

            # Get prompt (optionally enriched with news/economic context)
            prompt = self.config.get(
                "prompt_template",
                self._default_prompt(context_data=context_data),
            )

            # Save chart image for audit trail
            self._save_audit_chart(image_bytes, pair_name, timeframe, "analysis")

            # Call Gemini
            response = self._model.generate_content([prompt, image])

            if response and response.text:
                logger.info(
                    f"[Gemini] {pair_name} {timeframe} response: "
                    f"{response.text[:300]}"
                )
                result = self._parse_ai_response(response.text, df)
                if result.is_valid:
                    logger.info(
                        f"[Gemini] SIGNAL: {pair_name} {result.direction.value} "
                        f"pattern={result.pattern.value} conf={result.confidence}% "
                        f"entry={result.entry_price:.5f} sl={result.stop_loss:.5f} "
                        f"tp={result.take_profit:.5f} "
                        f"reason={result.reasoning}"
                    )
                else:
                    logger.info(f"[Gemini] {pair_name}: No signal detected")
                self._set_cache(cache_key, result)
                return result
            else:
                logger.warning("Empty response from Gemini")
                return PatternSignal(PatternType.NO_SIGNAL, SignalDirection.NONE, 0)

        except Exception as e:
            logger.error(f"AI chart analysis failed: {e}")
            return PatternSignal(PatternType.NO_SIGNAL, SignalDirection.NONE, 0)

    def _parse_ai_response(self, text: str, df: pd.DataFrame | None = None) -> PatternSignal:
        """Parse Gemini response into a PatternSignal.

        Args:
            text: Raw response text from Gemini.
            df: Optional DataFrame for price reference.

        Returns:
            PatternSignal parsed from AI response.
        """
        try:
            # Try to extract JSON from response
            json_str = text
            if "```json" in text:
                json_str = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                json_str = text.split("```")[1].split("```")[0]
            elif "{" in text:
                start = text.index("{")
                end = text.rindex("}") + 1
                json_str = text[start:end]

            data = json.loads(json_str.strip())

            signal_code = int(data.get("signal", 0))
            if signal_code == 0:
                return PatternSignal(PatternType.NO_SIGNAL, SignalDirection.NONE, 0)

            direction_str = data.get("direction", "NONE").upper()
            direction = SignalDirection.BUY if direction_str == "BUY" else \
                       SignalDirection.SELL if direction_str == "SELL" else \
                       SignalDirection.NONE

            pattern_name = data.get("pattern", "no_signal").lower().replace(" ", "_")
            pattern_map = {
                "double_bottom": PatternType.DOUBLE_BOTTOM,
                "double_top": PatternType.DOUBLE_TOP,
                "inverse_head_shoulders": PatternType.INVERSE_HEAD_SHOULDERS,
                "inverse_head_and_shoulders": PatternType.INVERSE_HEAD_SHOULDERS,
                "head_shoulders": PatternType.HEAD_SHOULDERS,
                "head_and_shoulders": PatternType.HEAD_SHOULDERS,
                "channel_breakout": PatternType.CHANNEL_BREAKOUT,
                "ma_crossover": PatternType.MA_CROSSOVER,
                "moving_average_crossover": PatternType.MA_CROSSOVER,
            }
            pattern = pattern_map.get(pattern_name, PatternType.NO_SIGNAL)

            key_levels = data.get("key_levels", {}) or {}
            current_price = float(df["close"].iloc[-1]) if df is not None and not df.empty else 0

            support = float(key_levels.get("support") or 0)
            resistance = float(key_levels.get("resistance") or 0)
            neckline = float(key_levels.get("neckline") or 0)

            # Use neckline as fallback if support/resistance missing
            if support == 0 and neckline > 0:
                support = neckline
            if resistance == 0 and neckline > 0:
                resistance = neckline

            if direction == SignalDirection.BUY:
                stop_loss = support if support > 0 else current_price * 0.99
                take_profit = resistance if resistance > 0 else current_price * 1.02
            else:
                stop_loss = resistance if resistance > 0 else current_price * 1.01
                take_profit = support if support > 0 else current_price * 0.98

            # Extract trend assessment from Gemini response
            trend_data = data.get("trend", {}) or {}
            trend_current = trend_data.get("current", "SIDEWAYS").upper()
            trend_reversal = bool(trend_data.get("reversal", False))
            trend_strength = trend_data.get("strength", "moderate").lower()
            trend_evidence = trend_data.get("evidence", "")

            if trend_current or trend_reversal:
                logger.info(
                    f"[Gemini] Trend: current={trend_current} "
                    f"reversal={trend_reversal} strength={trend_strength} "
                    f"evidence={trend_evidence[:80]}"
                )

            return PatternSignal(
                pattern=pattern,
                direction=direction,
                confidence=float(data.get("confidence", 50)),
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                neckline=neckline,
                support=support,
                resistance=resistance,
                reasoning=data.get("reasoning", "AI visual analysis"),
                extra={
                    "ai_source": "gemini",
                    "raw_response": data,
                    "trend_current": trend_current,
                    "trend_reversal": trend_reversal,
                    "trend_strength": trend_strength,
                    "trend_evidence": trend_evidence,
                },
            )

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to parse AI response: {e}")
            logger.debug(f"Raw response: {text[:500]}")
            return PatternSignal(PatternType.NO_SIGNAL, SignalDirection.NONE, 0)

    @staticmethod
    def _default_prompt(context_data: dict | None = None) -> str:
        """Default analysis prompt, optionally enriched with news/economic context."""
        # Build macro context section if data provided
        macro_section = ""
        if context_data:
            news_items: list = []
            for arts in (context_data.get("fx_news") or {}).values():
                news_items.extend(arts)
            news_items.sort(key=lambda a: a.get("age_hours", 99))
            news_lines = "\n".join(
                f"  - [{a['age_hours']}h ago] {a['title']}"
                for a in news_items[:4]
            ) or "  None"

            cal = context_data.get("economic_calendar") or []
            cal_lines = "\n".join(
                f"  - {e['date']} [{e['currency']}] {e['event']} "
                f"(forecast:{e.get('forecast','?')} prev:{e.get('previous','?')})"
                for e in cal[:4]
            ) or "  None"

            macro_section = f"""
MACRO CONTEXT (use this to strengthen or weaken your conviction):
Recent FX News:
{news_lines}

Upcoming Economic Events (next 3 days):
{cal_lines}

"""

        return f"""Analyze this FX chart image and answer TWO questions:
{macro_section}
QUESTION 1 - TREND ANALYSIS:
What is the current trend, and is it showing signs of reversal?
- Current trend direction (BULLISH / BEARISH / SIDEWAYS)
- Reversal signal present? (true / false)
- Reversal strength (weak / moderate / strong)
- Evidence: MA alignment, price structure (higher highs/lows or lower highs/lows), momentum

QUESTION 2 - PATTERN DETECTION:
Identify if any of these reversal/continuation patterns are present:
1. Double Bottom (W-shape) - especially "right-side up" variant with higher second low
2. Double Top (M-shape)
3. Inverse Head and Shoulders - especially "right shoulder up" variant
4. Head and Shoulders
5. Channel Breakout
6. Moving Average Crossover signals

If macro context is provided, factor in upcoming high-impact events that could affect direction.

Return ONLY a JSON object:
{{
  "signal": 1 or 0,
  "direction": "BUY" or "SELL" or "NONE",
  "pattern": "pattern_name",
  "confidence": 0-100,
  "trend": {{
    "current": "BULLISH" or "BEARISH" or "SIDEWAYS",
    "reversal": true or false,
    "strength": "weak" or "moderate" or "strong",
    "evidence": "brief description of MA alignment and price structure"
  }},
  "key_levels": {{"support": price, "resistance": price, "neckline": price}},
  "reasoning": "brief explanation combining trend, pattern, and macro context"
}}"""

    def analyze_market_regime(
        self,
        df: pd.DataFrame,
        pair_name: str = "",
    ) -> dict[str, Any]:
        """Analyze market regime using Gemini AI for adaptive risk.

        Determines current market conditions (trending, ranging, volatile)
        and returns risk adjustment recommendations.

        Args:
            df: OHLCV DataFrame with indicators.
            pair_name: Currency pair name.

        Returns:
            Dict with regime, volatility_level, trend_strength, risk_multiplier.
        """
        if not self._model:
            return self._fallback_regime_analysis(df)

        # Cache check
        cache_key = self._make_cache_key("regime", pair_name, "", df)
        cached = self._get_cache(cache_key)
        if cached is not None:
            logger.info(f"[Gemini] Cache hit: {pair_name} regime analysis")
            return cached

        try:
            # Generate chart image for regime analysis
            image_bytes = self.generate_chart_image(df, pair_name, "Regime Analysis")
            if image_bytes is None:
                return self._fallback_regime_analysis(df)

            # Save chart image for audit trail
            self._save_audit_chart(image_bytes, pair_name, "", "regime")

            image = Image.open(io.BytesIO(image_bytes))

            prompt = """Analyze this FX chart to determine the current MARKET REGIME.
Focus on:
1. Is the market TRENDING (clear direction) or RANGING (sideways)?
2. What is the VOLATILITY level? (candle sizes, Bollinger Band width)
3. How STRONG is the current trend? (moving average alignment)

Return ONLY a JSON object:
{
  "regime": "trending" or "ranging" or "volatile" or "uncertain",
  "volatility_level": "low" or "medium" or "high" or "extreme",
  "trend_strength": "weak" or "moderate" or "strong",
  "risk_multiplier": 0.0 to 1.0,
  "reasoning": "brief explanation"
}

Risk multiplier guidelines:
- Strong trend + low/medium volatility = 1.0 (full risk)
- Moderate trend = 0.8
- Ranging market = 0.6 (reduce risk, choppy conditions)
- High volatility + uncertain = 0.4
- Extreme volatility / news = 0.2 (minimal risk)"""

            response = self._model.generate_content([prompt, image])

            if response and response.text:
                result = self._parse_regime_response(response.text)
                self._set_cache(cache_key, result)
                return result
            else:
                return self._fallback_regime_analysis(df)

        except Exception as e:
            logger.error(f"AI regime analysis failed: {e}")
            return self._fallback_regime_analysis(df)

    # --------------------------------------------------
    # Cache helpers
    # --------------------------------------------------

    def _make_cache_key(
        self, kind: str, pair_name: str, timeframe: str, df: pd.DataFrame | None
    ) -> str:
        """Build a cache key from request parameters."""
        last_ts = ""
        if df is not None and not df.empty:
            last_ts = str(df.index[-1])
        raw = f"{kind}|{pair_name}|{timeframe}|{last_ts}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _get_cache(self, key: str) -> Any | None:
        """Return cached value if still valid, else None."""
        entry = self._cache.get(key)
        if entry and time.time() < entry["expires_at"]:
            return entry["result"]
        return None

    def _set_cache(self, key: str, result: Any) -> None:
        """Store result in cache with TTL."""
        self._cache[key] = {
            "result": result,
            "expires_at": time.time() + self._cache_ttl,
        }
        # Evict stale entries to prevent unbounded growth
        now = time.time()
        self._cache = {k: v for k, v in self._cache.items() if v["expires_at"] > now}

    def _parse_regime_response(self, text: str) -> dict[str, Any]:
        """Parse Gemini regime analysis response.

        Args:
            text: Raw response text.

        Returns:
            Parsed regime dict.
        """
        try:
            json_str = text
            if "```json" in text:
                json_str = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                json_str = text.split("```")[1].split("```")[0]
            elif "{" in text:
                start = text.index("{")
                end = text.rindex("}") + 1
                json_str = text[start:end]

            data = json.loads(json_str.strip())

            # Validate and clamp risk_multiplier
            risk_mult = float(data.get("risk_multiplier", 0.8))
            risk_mult = max(0.1, min(1.0, risk_mult))

            result = {
                "regime": data.get("regime", "normal"),
                "volatility_level": data.get("volatility_level", "medium"),
                "trend_strength": data.get("trend_strength", "moderate"),
                "risk_multiplier": risk_mult,
                "confidence": float(data.get("confidence", 50)),
                "reasoning": data.get("reasoning", ""),
            }

            logger.info(
                f"AI Regime: {result['regime']} | "
                f"Vol={result['volatility_level']} | "
                f"Trend={result['trend_strength']} | "
                f"Risk×{result['risk_multiplier']:.1f}"
            )
            return result

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to parse regime response: {e}")
            return {"regime": "normal", "volatility_level": "medium",
                    "trend_strength": "moderate", "risk_multiplier": 0.8,
                    "reasoning": "Parse failed, using conservative default"}

    def _fallback_regime_analysis(self, df: pd.DataFrame) -> dict[str, Any]:
        """Indicator-based regime analysis when AI is unavailable.

        Uses ATR, ADX, and Bollinger Band width to determine regime.

        Args:
            df: DataFrame with indicators.

        Returns:
            Regime dict based on technical indicators.
        """
        regime = "normal"
        volatility = "medium"
        trend_strength = "moderate"
        risk_mult = 0.8

        if df is None or df.empty:
            return {"regime": regime, "volatility_level": volatility,
                    "trend_strength": trend_strength, "risk_multiplier": risk_mult,
                    "reasoning": "No data available"}

        # ATR-based volatility detection
        if "atr" in df.columns:
            atr = float(df["atr"].iloc[-1])
            close = float(df["close"].iloc[-1])
            atr_pct = (atr / close) * 100 if close > 0 else 0

            if atr_pct > 2.0:
                volatility = "extreme"
            elif atr_pct > 1.0:
                volatility = "high"
            elif atr_pct > 0.5:
                volatility = "medium"
            else:
                volatility = "low"

        # ADX-based trend detection
        if "adx" in df.columns:
            adx = float(df["adx"].iloc[-1])
            if adx > 40:
                trend_strength = "strong"
                regime = "trending"
            elif adx > 25:
                trend_strength = "moderate"
                regime = "trending"
            else:
                trend_strength = "weak"
                regime = "ranging"

        # Calculate risk multiplier
        if regime == "trending" and volatility in ("low", "medium"):
            risk_mult = 1.0
        elif regime == "trending":
            risk_mult = 0.8
        elif regime == "ranging" and volatility == "low":
            risk_mult = 0.6
        elif volatility == "extreme":
            risk_mult = 0.3
        elif volatility == "high":
            risk_mult = 0.5
        else:
            risk_mult = 0.6

        reasoning = (
            f"Indicator-based: ADX→{trend_strength} trend, "
            f"ATR→{volatility} volatility"
        )

        logger.info(
            f"Fallback Regime: {regime} | Vol={volatility} | "
            f"Trend={trend_strength} | Risk×{risk_mult:.1f}"
        )

        return {
            "regime": regime,
            "volatility_level": volatility,
            "trend_strength": trend_strength,
            "risk_multiplier": risk_mult,
            "reasoning": reasoning,
        }

    def _save_audit_chart(
        self, image_bytes: bytes, pair_name: str, timeframe: str, analysis_type: str
    ) -> None:
        """Save chart image as audit trail for Gemini analysis.

        Args:
            image_bytes: PNG image bytes.
            pair_name: Currency pair name.
            timeframe: Chart timeframe.
            analysis_type: 'analysis' or 'regime'.
        """
        try:
            from datetime import datetime as dt
            now = dt.now()
            timestamp = now.strftime("%Y%m%d_%H%M%S")
            safe_pair = pair_name.replace("/", "_").replace(" ", "_")
            filename = f"{timestamp}_{safe_pair}_{analysis_type}.png"
            # 日付フォルダ（YYYY-MM）に整理して全期間永続保管
            date_folder = now.strftime("%Y-%m")
            output_dir = Path("data/gemini_audit") / date_folder
            output_dir.mkdir(parents=True, exist_ok=True)
            filepath = output_dir / filename
            with open(filepath, "wb") as f:
                f.write(image_bytes)
            logger.debug(f"[Gemini] Chart saved: {filepath}")
        except Exception as e:
            logger.debug(f"Audit chart save failed (non-critical): {e}")

    def evaluate_position(
        self,
        df: pd.DataFrame,
        pair_name: str,
        timeframe: str,
        position_direction: str,
        entry_price: float,
        current_price: float,
    ) -> dict[str, Any]:
        """Re-evaluate an open position: HOLD or CLOSE?  (Layer 2)

        Args:
            df: OHLCV DataFrame with indicators.
            pair_name: Currency pair name.
            timeframe: Chart timeframe string.
            position_direction: "BUY" or "SELL".
            entry_price: Original entry price.
            current_price: Current market price.

        Returns:
            {
              "action": "HOLD" | "CLOSE",
              "direction": SignalDirection,   # opposite = close signal
              "confidence": float,
              "reasoning": str,
            }
        """
        from src.strategies.pattern_detector import SignalDirection as SD
        _hold = {"action": "HOLD", "direction": SD.NONE, "confidence": 50.0, "reasoning": ""}

        if not self._model:
            return _hold

        cache_key = self._make_cache_key(
            f"eval_{position_direction}", pair_name, timeframe, df
        )
        cached = self._get_cache(cache_key)
        if cached is not None:
            logger.info(f"[Gemini Eval] Cache hit: {pair_name} position evaluation")
            return cached

        try:
            image_bytes = self.generate_chart_image(df, pair_name, timeframe)
            if image_bytes is None:
                return _hold

            pnl_pct = (current_price - entry_price) / entry_price * 100
            if position_direction == "SELL":
                pnl_pct = -pnl_pct

            image = Image.open(io.BytesIO(image_bytes))
            prompt = f"""You are reviewing an OPEN FX POSITION.

POSITION:
- Pair: {pair_name}
- Direction: {position_direction}
- Entry: {entry_price:.5f}
- Current price: {current_price:.5f}
- Unrealized P&L: {pnl_pct:+.2f}%

QUESTION: Based on the current chart, should this position be HELD or CLOSED?

Criteria for CLOSE recommendation:
1. The original trade pattern has clearly failed or invalidated
2. Strong reversal signals have formed against the position
3. Price structure has fundamentally changed

Only recommend CLOSE if confidence >= 80 with clear reversal evidence.
When uncertain, choose HOLD.

Return ONLY a JSON object:
{{
  "action": "HOLD" or "CLOSE",
  "confidence": 0-100,
  "direction": "BUY" or "SELL" (direction of the reversal signal if CLOSE, else same as position),
  "reasoning": "max 100 chars"
}}"""

            response = self._model.generate_content([prompt, image])
            if not (response and response.text):
                return _hold

            result = self._parse_eval_response(response.text, position_direction)
            logger.info(
                f"[Gemini Eval] {pair_name} {position_direction}: "
                f"action={result['action']} conf={result['confidence']:.0f}% "
                f"reason={result['reasoning'][:60]}"
            )
            self._set_cache(cache_key, result)
            return result

        except Exception as e:
            logger.error(f"[Gemini] evaluate_position failed: {e}")
            return _hold

    def _parse_eval_response(self, text: str, position_direction: str) -> dict[str, Any]:
        """Parse Gemini position evaluation response."""
        from src.strategies.pattern_detector import SignalDirection as SD
        try:
            json_str = text
            if "```json" in text:
                json_str = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                json_str = text.split("```")[1].split("```")[0]
            elif "{" in text:
                json_str = text[text.index("{"):text.rindex("}") + 1]

            data = json.loads(json_str.strip())
            action = data.get("action", "HOLD").upper()
            confidence = float(data.get("confidence", 50))
            dir_str = data.get("direction", position_direction).upper()
            direction = (
                SD.BUY if dir_str == "BUY"
                else SD.SELL if dir_str == "SELL"
                else SD.NONE
            )
            return {
                "action": action,
                "direction": direction,
                "confidence": confidence,
                "reasoning": str(data.get("reasoning", "")),
            }
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"[Gemini Eval] Parse failed: {e}")
            from src.strategies.pattern_detector import SignalDirection as SD
            return {"action": "HOLD", "direction": SD.NONE, "confidence": 50.0, "reasoning": ""}

    def save_chart(self, image_bytes: bytes, filename: str, output_dir: str = "data/charts") -> str:
        """Save chart image to file.

        Args:
            image_bytes: PNG image bytes.
            filename: Output filename.
            output_dir: Output directory.

        Returns:
            Path to saved file.
        """
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        filepath = path / filename
        with open(filepath, "wb") as f:
            f.write(image_bytes)
        return str(filepath)
