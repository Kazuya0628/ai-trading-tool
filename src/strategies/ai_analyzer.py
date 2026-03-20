"""AI Visual Chart Analysis using Google Gemini.

Implements AI Sennin's approach:
1. Generate chart image
2. Send to AI (Gemini) for pattern recognition
3. Get binary signal (1=trade, 0=no signal)
4. Execute based on AI classification
"""

from __future__ import annotations

import io
import json
import tempfile
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

            # Chart settings
            width = self.config.get("chart_width", 1200)
            height = self.config.get("chart_height", 800)
            style = self.config.get("chart_style", "charles")

            kwargs = {
                "type": "candle",
                "style": style,
                "title": f"{pair_name} {timeframe}" if pair_name else "",
                "figsize": (width / 100, height / 100),
                "volume": "Volume" in chart_df.columns and chart_df["Volume"].sum() > 0,
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

        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_bytes))

            # Get prompt
            prompt = self.config.get("prompt_template", self._default_prompt())

            # Call Gemini
            response = self._model.generate_content([prompt, image])

            if response and response.text:
                return self._parse_ai_response(response.text, df)
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

            key_levels = data.get("key_levels", {})
            current_price = float(df["close"].iloc[-1]) if df is not None and not df.empty else 0

            return PatternSignal(
                pattern=pattern,
                direction=direction,
                confidence=float(data.get("confidence", 50)),
                entry_price=current_price,
                stop_loss=float(key_levels.get("support", 0)),
                take_profit=float(key_levels.get("resistance", 0)) if direction == SignalDirection.BUY
                           else float(key_levels.get("support", 0)),
                neckline=float(key_levels.get("neckline", 0)),
                support=float(key_levels.get("support", 0)),
                resistance=float(key_levels.get("resistance", 0)),
                reasoning=data.get("reasoning", "AI visual analysis"),
                extra={"ai_source": "gemini", "raw_response": data},
            )

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to parse AI response: {e}")
            logger.debug(f"Raw response: {text[:500]}")
            return PatternSignal(PatternType.NO_SIGNAL, SignalDirection.NONE, 0)

    @staticmethod
    def _default_prompt() -> str:
        """Default analysis prompt if not configured."""
        return """Analyze this FX chart image. Identify if any of these patterns are present:
1. Double Bottom (W-shape) - especially "right-side up" variant with higher second low
2. Double Top (M-shape)
3. Inverse Head and Shoulders - especially "right shoulder up" variant
4. Head and Shoulders
5. Channel Breakout
6. Moving Average Crossover signals

Return ONLY a JSON object:
{
  "signal": 1 or 0,
  "direction": "BUY" or "SELL" or "NONE",
  "pattern": "pattern_name",
  "confidence": 0-100,
  "key_levels": {"support": price, "resistance": price, "neckline": price},
  "reasoning": "brief explanation"
}"""

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
