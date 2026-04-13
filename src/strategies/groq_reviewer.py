"""Groq AI による相場レジーム判断・週次戦略レビュー。

役割分担:
  Gemini  → チャート画像分析（シグナル発生時のみ・画像必須）
  Groq    → 相場レジーム判断（毎サイクル・指標＋外部センチメント）
            週次パフォーマンスレビュー・閾値調整判断
"""

from __future__ import annotations

import os
from typing import Any

from loguru import logger


class GroqReviewer:
    """Groq / Llama 3.3 を使った週次戦略レビュー。"""

    MODEL = "llama-3.3-70b-versatile"

    def __init__(
        self,
        *,
        client: Any | None = None,
        daily_limit: int | None = None,
    ) -> None:
        self._daily_count: int = 0
        self._daily_count_utc_date: str = self._utc_today()
        self._daily_limit: int = self._resolve_daily_limit(daily_limit)
        # Daily quota circuit breaker (Groq free tier TPD limit).
        # When 429 hits, suspend until next UTC day.
        self._rate_limited_date: str = ""

        if client is not None:
            self._client = client
            logger.info("[Groq] GroqReviewer initialized with injected client")
            return

        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            logger.warning("[Groq] GROQ_API_KEY が未設定です。週次レビューはスキップされます。")
            self._client = None
            return
        try:
            from groq import Groq
            self._client = Groq(api_key=api_key)
            logger.info("[Groq] GroqReviewer 初期化完了")
        except ImportError:
            logger.warning("[Groq] groq パッケージがインストールされていません。pip install groq")
            self._client = None

    @staticmethod
    def _utc_today() -> str:
        from datetime import datetime as _dt

        return _dt.utcnow().strftime("%Y-%m-%d")

    @staticmethod
    def _resolve_daily_limit(configured_limit: int | None) -> int:
        if configured_limit is not None:
            try:
                return max(1, int(configured_limit))
            except (TypeError, ValueError):
                pass

        env_limit = os.getenv("GROQ_DAILY_HARD_LIMIT", "")
        try:
            return max(1, int(env_limit)) if env_limit else 9999
        except ValueError:
            return 9999

    def _refresh_daily_counter(self) -> None:
        today = self._utc_today()
        if self._daily_count_utc_date != today:
            self._daily_count = 0
            self._daily_count_utc_date = today

    def _consume_daily_call(self) -> None:
        self._refresh_daily_counter()
        self._daily_count += 1

    @property
    def daily_count(self) -> int:
        self._refresh_daily_counter()
        return self._daily_count

    @property
    def daily_limit(self) -> int:
        return self._daily_limit

    @property
    def daily_remaining(self) -> int:
        return max(0, self.daily_limit - self.daily_count)

    @property
    def available(self) -> bool:
        if self._client is None:
            return False
        self._refresh_daily_counter()
        # Auto-reset on new day
        today = self._utc_today()
        if self._rate_limited_date and self._rate_limited_date != today:
            self._rate_limited_date = ""
            logger.info("[Groq] Daily quota reset — resuming Groq calls")
        if self._rate_limited_date:
            return False
        return self._daily_count < self._daily_limit

    def get_status(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "daily_used": self.daily_count,
            "daily_hard_limit": self.daily_limit,
            "daily_remaining": self.daily_remaining,
        }

    def _handle_rate_limit(self, err: Exception) -> bool:
        """Return True if the error is a 429 rate limit and breaker was tripped."""
        msg = str(err)
        if "429" in msg or "rate_limit" in msg.lower():
            self._rate_limited_date = self._utc_today()
            logger.warning(
                "[Groq] Daily token quota exhausted — suspending Groq until tomorrow (UTC)"
            )
            return True
        return False

    def review_weekly_performance(
        self,
        trades: list[dict],
        current_threshold: int,
        proposed_threshold: int,
        market_regime: str = "不明",
    ) -> dict[str, Any]:
        """直近トレード実績を分析し、閾値変更の妥当性をAIが判断する。

        Args:
            trades: 直近トレードのリスト（pnl, pattern, direction, exit_reason を含む）
            current_threshold: 現在の信頼度閾値（%）
            proposed_threshold: ルールが提案する新閾値（%）
            market_regime: 現在の相場環境（Geminiが判定した値）

        Returns:
            {
              "approved": bool,         # 変更を承認するか
              "final_threshold": int,   # 最終的な推奨閾値
              "reasoning": str,         # AIの判断理由
              "warnings": list[str],    # 注意事項
            }
        """
        if not self.available:
            return {
                "approved": True,
                "final_threshold": proposed_threshold,
                "reasoning": "Groq未設定のためルール判断をそのまま採用",
                "warnings": [],
            }

        try:
            summary = self._build_trade_summary(trades)
            prompt = self._build_prompt(
                summary, current_threshold, proposed_threshold, market_regime
            )
            response = self._client.chat.completions.create(
                model=self.MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "あなたはFXトレードシステムの戦略アドバイザーです。"
                            "提供されたパフォーマンスデータを分析し、"
                            "信頼度閾値の変更が適切かどうかを判断してください。"
                            "必ずJSON形式で回答してください。"
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=512,
                temperature=0.2,
            )
            self._consume_daily_call()
            raw = response.choices[0].message.content.strip()
            return self._parse_response(raw, proposed_threshold)

        except Exception as e:
            logger.error(f"[Groq] 週次レビュー失敗: {e}")
            return {
                "approved": True,
                "final_threshold": proposed_threshold,
                "reasoning": f"Groq呼び出し失敗のためルール判断を採用: {e}",
                "warnings": [],
            }

    def _build_trade_summary(self, trades: list[dict]) -> dict:
        """トレードリストから集計サマリーを作成する。"""
        if not trades:
            return {"total": 0}

        pnls = [t.get("pnl") or 0.0 for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        pf = gross_profit / gross_loss if gross_loss > 0 else 999.0

        # パターン別勝率
        pattern_stats: dict[str, dict] = {}
        for t in trades:
            pat = t.get("pattern", "unknown")
            if pat not in pattern_stats:
                pattern_stats[pat] = {"wins": 0, "total": 0}
            pattern_stats[pat]["total"] += 1
            if (t.get("pnl") or 0) > 0:
                pattern_stats[pat]["wins"] += 1

        pattern_summary = {
            k: f"{v['wins']}/{v['total']} ({v['wins']/v['total']*100:.0f}%)"
            for k, v in pattern_stats.items()
        }

        # 連敗カウント
        max_streak = 0
        streak = 0
        for p in pnls:
            if p < 0:
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 0

        return {
            "total": len(trades),
            "win_rate": f"{len(wins)/len(trades)*100:.1f}%",
            "profit_factor": round(pf, 2),
            "total_pnl": round(sum(pnls), 0),
            "avg_win": round(sum(wins) / len(wins), 0) if wins else 0,
            "avg_loss": round(sum(losses) / len(losses), 0) if losses else 0,
            "max_consecutive_losses": max_streak,
            "pattern_stats": pattern_summary,
        }

    def _build_prompt(
        self,
        summary: dict,
        current_thr: int,
        proposed_thr: int,
        regime: str,
    ) -> str:
        direction = "引き上げ" if proposed_thr > current_thr else "引き下げ"
        return f"""
## FXトレードシステム 週次レビュー

### 直近{summary.get('total', 0)}トレードの実績
- 勝率: {summary.get('win_rate', 'N/A')}
- プロフィットファクター: {summary.get('profit_factor', 'N/A')}
- 累積損益: ¥{summary.get('total_pnl', 0):,.0f}
- 平均利益: ¥{summary.get('avg_win', 0):,.0f}
- 平均損失: ¥{summary.get('avg_loss', 0):,.0f}
- 最大連敗: {summary.get('max_consecutive_losses', 0)}回
- パターン別勝率: {summary.get('pattern_stats', {})}

### 現在の相場環境
{regime}

### 提案されている変更
現在の信頼度閾値: {current_thr}%
提案の閾値: {proposed_thr}% ({direction})

### 判断してください
以下のJSON形式のみで回答してください（説明文は不要）:
{{
  "approved": true または false,
  "final_threshold": 整数（50〜65の範囲で推奨閾値）,
  "reasoning": "判断理由を100字以内で",
  "warnings": ["注意事項があれば配列で、なければ空配列"]
}}
"""

    def _parse_response(self, raw: str, fallback: int) -> dict[str, Any]:
        """Groqの返答からJSONを抽出してパースする。"""
        import json
        import re

        # JSONブロックを抽出
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            logger.warning(f"[Groq] JSON抽出失敗: {raw[:200]}")
            return {
                "approved": True,
                "final_threshold": fallback,
                "reasoning": "パース失敗のためルール判断を採用",
                "warnings": [],
            }
        try:
            data = json.loads(match.group())
            return {
                "approved": bool(data.get("approved", True)),
                "final_threshold": int(data.get("final_threshold", fallback)),
                "reasoning": str(data.get("reasoning", "")),
                "warnings": list(data.get("warnings", [])),
            }
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"[Groq] JSONパース失敗: {e}")
            return {
                "approved": True,
                "final_threshold": fallback,
                "reasoning": f"パース失敗: {e}",
                "warnings": [],
            }

    # ──────────────────────────────────────────────────
    # 相場レジーム分析（Geminiの画像分析を代替）
    # ──────────────────────────────────────────────────

    def analyze_market_regime(
        self,
        indicators: dict[str, Any],
        sentiment_data: dict[str, Any],
        pair_name: str = "",
    ) -> dict[str, Any]:
        """テクニカル指標＋外部センチメントから相場レジームを判断する。

        Args:
            indicators: {"atr": float, "adx": float, "rsi": float,
                         "price": float, "ma20": float, "ma200": float,
                         "bb_width": float}
            sentiment_data: SentimentFetcher.fetch_all() の返り値
            pair_name: "EUR/USD" など

        Returns:
            {
              "regime": "trending" | "ranging" | "high_volatility" | "caution",
              "volatility_level": "low" | "medium" | "high",
              "trend_strength": "weak" | "moderate" | "strong",
              "risk_multiplier": float,  # 0.5〜1.2
              "confidence": int,         # 0〜100
              "reasoning": str,
            }
        """
        if not self.available:
            return self._default_regime()

        try:
            prompt = self._build_regime_prompt(indicators, sentiment_data, pair_name)
            response = self._client.chat.completions.create(
                model=self.MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "あなたはFX市場のプロアナリストです。"
                            "提供されたテクニカル指標・ニュース・経済指標・SNS情報をもとに"
                            "現在の相場レジームを判断してください。"
                            "必ずJSON形式のみで回答してください。"
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=400,
                temperature=0.1,
            )
            self._consume_daily_call()
            raw = response.choices[0].message.content.strip()
            result = self._parse_regime_response(raw)
            logger.info(
                f"[Groq レジーム] {pair_name}: {result['regime']} / "
                f"ボラ={result['volatility_level']} / "
                f"トレンド={result['trend_strength']} / "
                f"リスク倍率={result['risk_multiplier']}"
            )
            return result

        except Exception as e:
            if not self._handle_rate_limit(e):
                logger.error(f"[Groq] レジーム分析失敗: {e}")
            return self._default_regime()

    def _build_regime_prompt(
        self,
        indicators: dict[str, Any],
        sentiment_data: dict[str, Any],
        pair_name: str,
    ) -> str:
        """相場レジーム判断用のプロンプトを構築する。"""
        # テクニカル指標
        atr = indicators.get("atr", "N/A")
        adx = indicators.get("adx", "N/A")
        rsi = indicators.get("rsi", "N/A")
        price = indicators.get("price", "N/A")
        ma20 = indicators.get("ma20", "N/A")
        ma200 = indicators.get("ma200", "N/A")
        bb_width = indicators.get("bb_width", "N/A")

        price_vs_ma = "N/A"
        try:
            if float(price) > float(ma200):
                price_vs_ma = f"MA200上方（+{(float(price)/float(ma200)-1)*100:.1f}%）"
            else:
                price_vs_ma = f"MA200下方（{(float(price)/float(ma200)-1)*100:.1f}%）"
        except (TypeError, ValueError, ZeroDivisionError):
            pass

        # 経済指標（今後3日）
        cal = sentiment_data.get("economic_calendar", [])
        cal_text = "\n".join(
            f"  - {e['date']} {e['time']} [{e['currency']}] {e['event']} "
            f"(予測:{e['forecast']} 前回:{e['previous']})"
            for e in cal[:5]
        ) or "  なし"

        # FXニュース
        news_items = sentiment_data.get("fx_news", {})
        all_news = []
        for arts in news_items.values():
            all_news.extend(arts)
        all_news.sort(key=lambda a: a.get("age_hours", 99))
        news_text = "\n".join(
            f"  - [{a['age_hours']}h前] {a['title']}"
            for a in all_news[:5]
        ) or "  なし"

        # Reddit（用心発言）
        reddit = sentiment_data.get("reddit", [])
        reddit_text = "\n".join(
            f"  - [score={p['score']}] {p['title']}"
            for p in reddit[:4]
        ) or "  なし"

        return f"""
## 相場レジーム分析: {pair_name}

### テクニカル指標
- ATR: {atr}
- ADX: {adx}（25以上＝トレンド、25未満＝レンジ）
- RSI: {rsi}
- 価格 vs MA200: {price_vs_ma}
- BB幅: {bb_width}

### 直近FXニュース
{news_text}

### 今後3日間の高インパクト経済指標
{cal_text}

### Reddit r/Forex（用心発言・市場センチメント）
{reddit_text}

### 判断してください
以下のJSON形式のみで回答してください:
{{
  "regime": "trending" または "ranging" または "high_volatility" または "caution",
  "volatility_level": "low" または "medium" または "high",
  "trend_strength": "weak" または "moderate" または "strong",
  "risk_multiplier": 0.5〜1.2の数値（通常=1.0、高リスク=0.5〜0.8、好環境=1.0〜1.2）,
  "confidence": 0〜100の整数,
  "reasoning": "判断理由を80字以内で"
}}
"""

    def _parse_regime_response(self, raw: str) -> dict[str, Any]:
        """Groqのレジーム判断レスポンスをパースする。"""
        import json
        import re

        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return self._default_regime()
        try:
            data = json.loads(match.group())
            return {
                "regime": data.get("regime", "ranging"),
                "volatility_level": data.get("volatility_level", "medium"),
                "trend_strength": data.get("trend_strength", "moderate"),
                "risk_multiplier": float(data.get("risk_multiplier", 1.0)),
                "confidence": int(data.get("confidence", 50)),
                "reasoning": str(data.get("reasoning", "")),
            }
        except (json.JSONDecodeError, ValueError):
            return self._default_regime()

    @staticmethod
    def _default_regime() -> dict[str, Any]:
        return {
            "regime": "ranging",
            "volatility_level": "medium",
            "trend_strength": "moderate",
            "risk_multiplier": 1.0,
            "confidence": 50,
            "reasoning": "Groq未使用のためデフォルト値",
        }

    # ──────────────────────────────────────────────────
    # 方向性投票（コンセンサスエンジン用）
    # ──────────────────────────────────────────────────

    def get_directional_vote(
        self,
        pair_name: str,
        indicators: dict[str, Any],
        sentiment_data: dict[str, Any],
        current_regime: str = "",
    ) -> dict[str, Any]:
        """ニュース・経済指標・指標から方向性を投票する。（Layer 3コンセンサス用）

        Args:
            pair_name: "USD/JPY" など。
            indicators: ATR / ADX / RSI / price / MA 辞書。
            sentiment_data: SentimentFetcher.fetch_all() の返り値。
            current_regime: Groqが直前に判断したレジーム文字列。

        Returns:
            {
              "direction": SignalDirection,   # BUY / SELL / NONE
              "confidence": float,            # 0-100
              "reasoning": str,
            }
        """
        from src.strategies.pattern_detector import SignalDirection as SD
        _neutral = {"direction": SD.NONE, "confidence": 50.0, "reasoning": "Groq未使用"}

        if not self.available:
            return _neutral

        try:
            prompt = self._build_direction_prompt(
                pair_name, indicators, sentiment_data, current_regime
            )
            response = self._client.chat.completions.create(
                model=self.MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "あなたはFXトレードシステムのAIアナリストです。"
                            "テクニカル指標・ニュース・経済指標をもとに今後の方向性を判断し、"
                            "必ずJSON形式のみで回答してください。"
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=256,
                temperature=0.1,
            )
            self._consume_daily_call()
            raw = response.choices[0].message.content.strip()
            result = self._parse_direction_response(raw)
            logger.info(
                f"[Groq Vote] {pair_name}: {result['direction'].value} "
                f"conf={result['confidence']:.0f}% reason={result['reasoning'][:50]}"
            )
            return result

        except Exception as e:
            if not self._handle_rate_limit(e):
                logger.error(f"[Groq] 方向性投票失敗: {e}")
            return _neutral

    def _build_direction_prompt(
        self,
        pair_name: str,
        indicators: dict[str, Any],
        sentiment_data: dict[str, Any],
        regime: str,
    ) -> str:
        rsi = indicators.get("rsi", "N/A")
        adx = indicators.get("adx", "N/A")
        price = indicators.get("price", "N/A")
        ma20 = indicators.get("ma20", "N/A")
        ma200 = indicators.get("ma200", "N/A")

        cal = sentiment_data.get("economic_calendar", [])
        cal_text = "\n".join(
            f"  - {e['date']} [{e['currency']}] {e['event']}"
            f"(予測:{e.get('forecast','?')} 前回:{e.get('previous','?')})"
            for e in cal[:4]
        ) or "  なし"

        news_items: list = []
        for arts in (sentiment_data.get("fx_news") or {}).values():
            news_items.extend(arts)
        news_items.sort(key=lambda a: a.get("age_hours", 99))
        news_text = "\n".join(
            f"  - [{a['age_hours']}h前] {a['title']}"
            for a in news_items[:4]
        ) or "  なし"

        return f"""
## {pair_name} の方向性判断

### テクニカル指標
- RSI: {rsi} | ADX: {adx}
- 価格: {price} (MA20: {ma20} / MA200: {ma200})
- 現在レジーム: {regime}

### 直近FXニュース
{news_text}

### 今後の経済指標
{cal_text}

### 判断
今後の{pair_name}の方向性をJSON形式のみで回答してください:
{{
  "direction": "BUY" または "SELL" または "NEUTRAL",
  "confidence": 0〜100,
  "reasoning": "理由を60字以内で"
}}
"""

    def _parse_direction_response(self, raw: str) -> dict[str, Any]:
        import json as _json
        import re
        from src.strategies.pattern_detector import SignalDirection as SD

        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return {"direction": SD.NONE, "confidence": 50.0, "reasoning": "パース失敗"}
        try:
            data = _json.loads(match.group())
            dir_str = data.get("direction", "NEUTRAL").upper()
            direction = (
                SD.BUY if dir_str == "BUY"
                else SD.SELL if dir_str == "SELL"
                else SD.NONE
            )
            return {
                "direction": direction,
                "confidence": float(data.get("confidence", 50)),
                "reasoning": str(data.get("reasoning", "")),
            }
        except (ValueError, KeyError):
            return {"direction": SD.NONE, "confidence": 50.0, "reasoning": "パース失敗"}
