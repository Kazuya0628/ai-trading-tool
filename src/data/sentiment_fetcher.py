"""外部センチメント・経済指標データの取得。

データソース（すべて無料・APIキー不要）:
  - yfinance ニュース : FXペア関連の最新ニュース見出し
  - ForexFactory      : 経済指標カレンダー（高インパクトイベント）
  - Reddit r/Forex    : トレーダーの議論・用心発言
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import requests
from loguru import logger


class SentimentFetcher:
    """外部センチメント・経済指標を取得するクラス。"""

    FOREXFACTORY_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
    REDDIT_URL = "https://www.reddit.com/r/Forex/new.json"

    # 通貨ペア → Yahoo Financeティッカー
    YAHOO_TICKER_MAP = {
        "USD_JPY": "USDJPY=X",
        "EUR_USD": "EURUSD=X",
        "GBP_JPY": "GBPJPY=X",
        "GBP_USD": "GBPUSD=X",
        "EUR_JPY": "EURJPY=X",
    }

    TARGET_CURRENCIES = {"USD", "EUR", "JPY", "GBP"}
    HIGH_IMPACT = {"High"}

    def __init__(self, timeout: int = 10) -> None:
        self._timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        })

    # ──────────────────────────────────────────────────
    # yfinance ニュース（StockTwits代替）
    # ──────────────────────────────────────────────────

    def fetch_fx_news(self, instruments: list[str]) -> dict[str, Any]:
        """yfinanceからFXペア関連ニュースを取得する。

        Returns:
            {
              "EUR_USD": [{"title": "...", "publisher": "...", "age_hours": 2.1}, ...],
              ...
            }
        """
        try:
            import yfinance as yf
        except ImportError:
            logger.warning("[News] yfinance未インストール")
            return {}

        results: dict[str, Any] = {}
        for inst in instruments:
            ticker_sym = self.YAHOO_TICKER_MAP.get(inst)
            if not ticker_sym:
                continue
            try:
                ticker = yf.Ticker(ticker_sym)
                news_items = ticker.news or []
                now_ts = datetime.now(timezone.utc).timestamp()

                articles = []
                for item in news_items[:8]:
                    # 新形式: {"id": ..., "content": {...}}
                    content = item.get("content") or {}
                    title = content.get("title") or item.get("title", "")
                    publisher = (
                        (content.get("provider") or {}).get("displayName", "")
                        or item.get("publisher", "")
                    )
                    # 公開日時（ISO8601 または unix タイムスタンプ）
                    pub_date = content.get("pubDate") or content.get("displayTime", "")
                    pub_ts = item.get("providerPublishTime", 0)
                    if pub_date and not pub_ts:
                        try:
                            pub_ts = datetime.fromisoformat(
                                pub_date.replace("Z", "+00:00")
                            ).timestamp()
                        except ValueError:
                            pass
                    age_h = round((now_ts - pub_ts) / 3600, 1) if pub_ts else 99
                    if age_h > 48 or not title:
                        continue
                    articles.append({
                        "title": title[:120],
                        "publisher": publisher,
                        "age_hours": age_h,
                    })

                if articles:
                    results[inst] = articles
                    logger.debug(f"[News] {inst}: {len(articles)}件のニュース取得")

            except Exception as e:
                logger.warning(f"[News] {inst} 取得失敗: {e}")

        return results

    # ──────────────────────────────────────────────────
    # ForexFactory 経済指標カレンダー
    # ──────────────────────────────────────────────────

    def fetch_economic_calendar(self, days_ahead: int = 3) -> list[dict]:
        """ForexFactoryから今後N日間の高インパクト経済指標を取得する。

        Returns:
            [
              {
                "date": "2026-03-24",
                "time": "08:30",
                "currency": "USD",
                "event": "Non-Farm Payrolls",
                "impact": "High",
                "forecast": "180K",
                "previous": "151K",
              },
              ...
            ]
        """
        try:
            resp = self._session.get(self.FOREXFACTORY_URL, timeout=self._timeout)
            if resp.status_code != 200:
                logger.warning(f"[ForexFactory] HTTP {resp.status_code}")
                return []

            raw_events = resp.json()
            now = datetime.now(timezone.utc)
            cutoff = now + timedelta(days=days_ahead)

            events = []
            for ev in raw_events:
                currency = ev.get("country", "").upper()
                impact = ev.get("impact", "")

                if currency not in self.TARGET_CURRENCIES:
                    continue
                if impact not in self.HIGH_IMPACT:
                    continue

                # ISO 8601 形式 "2026-03-24T08:30:00-04:00" をパース
                date_raw = ev.get("date", "")
                try:
                    # Python 3.9 では %z が ±HH:MM をサポートしないため手動処理
                    ev_dt = datetime.fromisoformat(date_raw)
                    if ev_dt.tzinfo is None:
                        ev_dt = ev_dt.replace(tzinfo=timezone.utc)
                except ValueError:
                    continue

                if ev_dt < now or ev_dt > cutoff:
                    continue

                # UTC変換して表示
                ev_utc = ev_dt.astimezone(timezone.utc)
                events.append({
                    "date": ev_utc.strftime("%Y-%m-%d"),
                    "time": ev_utc.strftime("%H:%M") + " UTC",
                    "currency": currency,
                    "event": ev.get("title", ""),
                    "impact": impact,
                    "forecast": ev.get("forecast") or "N/A",
                    "previous": ev.get("previous") or "N/A",
                })

            events.sort(key=lambda e: e["date"] + e["time"])
            logger.debug(f"[ForexFactory] 高インパクトイベント: {len(events)}件")
            return events

        except Exception as e:
            logger.warning(f"[ForexFactory] 取得失敗: {e}")
            return []

    # ──────────────────────────────────────────────────
    # Reddit r/Forex
    # ──────────────────────────────────────────────────

    def fetch_reddit_forex(self, limit: int = 10) -> list[dict]:
        """Reddit r/Forexの最新・人気投稿を取得する。

        Returns:
            [{"title": "...", "score": 42, "comments": 15}, ...]
        """
        try:
            resp = self._session.get(
                self.REDDIT_URL,
                params={"limit": limit, "sort": "new"},
                timeout=self._timeout,
            )
            if resp.status_code != 200:
                logger.warning(f"[Reddit] HTTP {resp.status_code}")
                return []

            children = resp.json().get("data", {}).get("children", [])
            now_ts = datetime.now(timezone.utc).timestamp()
            posts = []
            for child in children:
                d = child.get("data", {})
                created = d.get("created_utc", 0)
                age_h = round((now_ts - created) / 3600, 1) if created else 99
                posts.append({
                    "title": d.get("title", "")[:120],
                    "score": d.get("score", 0),
                    "comments": d.get("num_comments", 0),
                    "age_hours": age_h,
                })

            posts.sort(key=lambda p: p["score"], reverse=True)
            logger.debug(f"[Reddit] r/Forex: {len(posts)}件")
            return posts

        except Exception as e:
            logger.warning(f"[Reddit] 取得失敗: {e}")
            return []

    # ──────────────────────────────────────────────────
    # 統合取得
    # ──────────────────────────────────────────────────

    def fetch_all(self, instruments: list[str]) -> dict[str, Any]:
        """全データソースを一括取得してまとめて返す。"""
        logger.info("[Sentiment] 外部データ取得開始...")
        result = {
            "fx_news": self.fetch_fx_news(instruments),
            "economic_calendar": self.fetch_economic_calendar(days_ahead=3),
            "reddit": self.fetch_reddit_forex(limit=10),
            "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        logger.info(
            f"[Sentiment] 完了: ニュース={len(result['fx_news'])}ペア / "
            f"経済指標={len(result['economic_calendar'])}件 / "
            f"Reddit={len(result['reddit'])}件"
        )
        return result
