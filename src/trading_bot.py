"""Automated Trading Bot - Thin Orchestrator.

Delegates business logic to specialized services:
- Scheduler: Layer 0/1/2 timing
- StrategyEngine: Pattern detection + AI analysis
- ConsensusEngine: 3-voter / 2-voter fallback voting
- RiskManager: Position sizing, risk validation
- ExecutionService: Order submission
- PortfolioService: NAV/DD/Exposure
- ReconciliationService: OANDA/DB consistency
- GeminiBudgetService: API call budget management

"1 Person Hedge Fund" - AI Sennin
"""

from __future__ import annotations

import time
import traceback
from datetime import datetime, date
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

try:
    import jpholiday  # type: ignore
    _JPHOLIDAY_AVAILABLE = True
except ImportError:
    _JPHOLIDAY_AVAILABLE = False

from src.app.runtime_state import RuntimeState
from src.app.scheduler import Scheduler
from src.data.indicators import TechnicalAnalyzer
from src.data.market_data import MarketDataManager
from src.domain.enums import RunMode
from src.models.position_store import PositionStore
from src.models.risk_manager import RiskManager
from src.services.execution_service import ExecutionService
from src.services.gemini_budget_service import GeminiBudgetService
from src.services.portfolio_service import PortfolioService
from src.services.reconciliation_service import ReconciliationService
from src.strategies.ai_analyzer import AIChartAnalyzer
from src.strategies.pattern_detector import PatternDetector, SignalDirection
from src.strategies.strategy_engine import StrategyEngine, TradeSignal
from src.utils.config_loader import TradingConfig, create_broker
from src.utils.logger import log_trade, setup_logging
from src.utils.notifier import Notifier


class TradingBot:
    """Thin orchestrator — delegates to specialized services.

    Responsibilities:
    - Wire up services and components
    - Receive Layer 0/1/2 callbacks from Scheduler
    - Manage runtime state and mode transitions
    - Maintain cycle_log for dashboard
    """

    def __init__(self, config_path: str | None = None) -> None:
        """Initialize TradingBot with all components.

        Args:
            config_path: Path to trading_config.yaml.
        """
        # Load configuration
        self.config = TradingConfig(config_path)

        # Setup logging
        log_cfg = self.config.config.get("logging", {})
        setup_logging(
            level=log_cfg.get("level", "INFO"),
            log_file=log_cfg.get("file", "logs/trading.log"),
            trade_log=log_cfg.get("trade_log", "logs/trades.log"),
        )

        logger.info("=" * 60)
        logger.info("AI FX Trading Bot - Initializing")
        logger.info(f"Mode: {'LIVE' if self.config.is_live else 'PAPER'}")
        logger.info(f"Data Source: {self.config.data_source}")
        logger.info("=" * 60)

        # --- Core Infrastructure ---
        self.broker = create_broker(self.config.env)
        self.data_manager = MarketDataManager(self.broker)
        self.tech_analyzer = TechnicalAnalyzer(self.config.indicators)
        self.position_store = PositionStore()

        # --- Strategy Components ---
        self.pattern_detector = PatternDetector(self.config.pattern_recognition)

        self.ai_analyzer = None
        if self.config.ai_analysis.get("enabled") and self.config.env.get("GEMINI_API_KEY"):
            self.ai_analyzer = AIChartAnalyzer(
                api_key=self.config.env["GEMINI_API_KEY"],
                config=self.config.ai_analysis,
            )

        self.strategy_engine = StrategyEngine(
            pattern_detector=self.pattern_detector,
            ai_analyzer=self.ai_analyzer,
            config={
                "indicators": self.config.indicators,
                "consensus": self.config.consensus_config,
            },
        )

        # --- Services (design doc: separated from orchestrator) ---
        from src.strategies.groq_reviewer import GroqReviewer
        from src.data.sentiment_fetcher import SentimentFetcher
        self.groq_reviewer = GroqReviewer(
            daily_limit=self.config.groq_config.get("daily_hard_limit")
        )
        self.sentiment_fetcher = SentimentFetcher()
        self._last_sentiment: dict = {}
        self._last_sentiment_ts: float = 0.0

        self.risk_manager = RiskManager(self.config.risk_management)
        self.gemini_budget = GeminiBudgetService(
            self.config.ai_analysis.get("budget", {})
        )
        _max_currency_exposure = self.config.risk_management.get(
            "max_currency_exposure_count", 3
        )
        self.portfolio = PortfolioService(max_currency_exposure=_max_currency_exposure)
        self.reconciliation = ReconciliationService(self.broker, self.position_store)

        self.notifier = Notifier(
            discord_webhook=self.config.env.get("DISCORD_WEBHOOK_URL", ""),
            line_token=self.config.env.get("LINE_NOTIFY_TOKEN", ""),
            line_config={
                "channel_access_token": self.config.env.get("LINE_CHANNEL_ACCESS_TOKEN", ""),
                "user_id": self.config.env.get("LINE_USER_ID", ""),
                "imgbb_api_key": self.config.env.get("IMGBB_API_KEY", ""),
            },
        )

        self.execution = ExecutionService(
            broker=self.broker,
            position_store=self.position_store,
            notifier=self.notifier,
            config=self.config.execution,
        )

        # --- Runtime State (centralized) ---
        self.state = RuntimeState()
        self.scheduler = Scheduler()

        # Legacy state (kept for backward compatibility during transition)
        self._running = False
        self._open_trades: dict[str, list[dict[str, Any]]] = {}
        self._cycle_log: dict[str, Any] = {}
        # Aliased from state for backward compat in existing methods
        self._gemini_cache: dict[str, dict[str, Any]] = self.state.gemini_cache

        # Runtime tracking attributes
        self._dd_warned_levels: set[float] = set()
        self._prev_regime: str = ""
        self._last_summary_date: str = ""
        self._last_weekly_review_date: str = ""
        self._position_eval_counter: int = 0

        self._restore_open_trades()

    def _restore_open_trades(self) -> None:
        """Restore open positions on startup — OANDA API is source of truth."""
        use_oanda = hasattr(self.broker, "_use_oanda_api") and self.broker._use_oanda_api

        if use_oanda:
            # OANDA mode: fetch live positions from API, enrich with local AI metadata
            try:
                oanda_positions = self.broker.get_positions()
                for pos in oanda_positions:
                    instrument = pos["instrument"]
                    deal_id = pos["deal_id"]
                    # Enrich with AI metadata (pattern, confidence, etc.) from SQLite
                    meta = self.position_store.get_position_by_deal_id(deal_id) or {}
                    if instrument not in self._open_trades:
                        self._open_trades[instrument] = []
                    self._open_trades[instrument].append({
                        **pos,
                        "opened_at": pos.get("open_time", ""),
                        "pattern": meta.get("pattern", "UNKNOWN"),
                        "confidence": meta.get("confidence", 0),
                        "signal_id": meta.get("signal_id"),
                        "is_fallback": bool(meta.get("is_fallback", False)),
                    })
                total = sum(len(v) for v in self._open_trades.values())
                logger.info(
                    f"Restored {total} open positions from OANDA API: "
                    f"{list(self._open_trades.keys())}"
                )
            except Exception as e:
                logger.error(f"Failed to restore positions from OANDA: {e}")
            return

        # Paper mode fallback: restore from SQLite
        try:
            positions = self.position_store.get_open_positions()
            if not positions:
                return

            for pos in positions:
                instrument = pos["instrument"]
                if instrument not in self._open_trades:
                    self._open_trades[instrument] = []
                self._open_trades[instrument].append(pos)
                if hasattr(self.broker, "_paper_positions"):
                    deal_id = pos.get("deal_id", "")
                    self.broker._paper_positions[deal_id] = {
                        "epic": instrument,
                        "deal_id": deal_id,
                        "direction": pos.get("direction", ""),
                        "entry_price": pos.get("entry_price", 0),
                        "size": pos.get("size", 0),
                        "stop_loss": pos.get("stop_loss", 0),
                        "take_profit": pos.get("take_profit", 0),
                        "trailing_stop": False,
                        "opened_at": pos.get("opened_at", ""),
                    }

            # Restore deal_id counter
            if hasattr(self.broker, "_next_deal_id"):
                self.broker._next_deal_id = self.position_store.get_next_deal_id()

            total_restored = sum(len(v) for v in self._open_trades.values())
            logger.info(
                f"Restored {total_restored} open positions from DB: "
                f"{list(self._open_trades.keys())}"
            )
        except Exception as e:
            logger.error(f"Failed to restore positions from DB: {e}")

    def _has_open_position_in_log(self, instrument: str) -> bool:
        """Check if there is an open position for the given instrument."""
        if hasattr(self.broker, "_use_oanda_api") and self.broker._use_oanda_api:
            # OANDA mode: _open_trades is the in-memory mirror of OANDA state
            return instrument in self._open_trades and bool(self._open_trades[instrument])
        return self.position_store.has_open_position(instrument)

    # --------------------------------------------------
    # Gemini Budget Helpers (delegated to GeminiBudgetService)
    # --------------------------------------------------

    def _gemini_budget_ok(self) -> bool:
        """Check if Gemini daily budget allows another call."""
        ok, _ = self.gemini_budget.can_call()
        return ok

    def _use_gemini_budget(self) -> None:
        """Consume one Gemini API call from the budget."""
        self.gemini_budget.consume()

    @property
    def _gemini_daily_count(self) -> int:
        return self.gemini_budget.daily_count

    @property
    def _gemini_daily_limit(self) -> int:
        return self.gemini_budget.daily_limit

    @property
    def _next_4h_time(self) -> str:
        return self.scheduler.next_4h_display()

    @_next_4h_time.setter
    def _next_4h_time(self, value: str) -> None:
        pass  # Computed property, setter is no-op for compatibility

    def _should_use_gemini(self, instrument: str, signals: list) -> bool:
        """Pre-filter: Geminiを呼ぶ価値があるか判定。"""
        return bool(signals)

    # --------------------------------------------------
    # Main Trading Loop (Scheduler-driven)
    # --------------------------------------------------

    def start(self) -> None:
        """Start the automated trading loop using Scheduler."""
        logger.info("Starting trading bot...")

        # Connect to OANDA
        if not self.broker.connect():
            logger.error("Failed to connect to OANDA. Aborting.")
            self.state.enter_analysis_only("OANDA connection failed at startup")
            return

        self._running = True
        self.state.running = True
        self.state.run_mode = RunMode.PAPER

        # Initial account info and portfolio state
        account = self.broker.get_account_info()
        if account:
            logger.info(
                f"Account: {account.get('account_name', 'N/A')} | "
                f"Balance: ¥{account.get('balance', 0):,.0f} | "
                f"Available: ¥{account.get('available', 0):,.0f}"
            )
            self.risk_manager.state.peak_balance = account.get("balance", 0)
            self.portfolio.update_from_broker(account)

        # Startup reconciliation
        recon_report = self.reconciliation.reconcile_on_startup()
        if recon_report.get("errors"):
            logger.warning(f"[Startup] Reconciliation had errors: {recon_report['errors']}")

        self.notifier.alert(
            f"Trading bot started.\n"
            f"Mode: {'LIVE' if self.config.is_live else 'PAPER'}\n"
            f"Pairs: {', '.join(self.config.active_pairs)}",
            line=False,
        )

        try:
            self.scheduler.run_loop(
                on_layer0=self._layer0_monitoring,
                on_layer1=self._layer1_entry_analysis,
                on_layer2=self._layer2_position_management,
                on_daily_reset=self._daily_reset,
                on_weekly_review=self._weekly_threshold_review,
                on_reconciliation=lambda: self.reconciliation.reconcile_periodic(),
                check_running=lambda: self._running,
                sleep_seconds=10,
            )
        except KeyboardInterrupt:
            logger.info("Bot stopped by user (Ctrl+C)")
        except Exception as e:
            logger.error(f"Fatal error: {e}")
            logger.error(traceback.format_exc())
            self.notifier.alert(f"FATAL ERROR: {e}")
        finally:
            self.stop()

    # --------------------------------------------------
    # Layer Callbacks (Scheduler-driven)
    # --------------------------------------------------

    def _layer0_monitoring(self) -> None:
        """Layer 0: Lightweight monitoring (every 5 min).

        - Quick analysis without Gemini
        - Update cycle_log for dashboard
        - No new entries allowed
        """
        try:
            self._run_cycle_inner(is_4h_candle=False)
        finally:
            self._write_ai_log()

    def _layer1_entry_analysis(self) -> None:
        """Layer 1: Full analysis at 4H close.

        - Gemini analysis (budget permitting)
        - Groq direction vote refresh
        - Consensus voting
        - New entries allowed
        """
        self.gemini_budget.start_new_cycle(self.scheduler.current_4h_slot())
        try:
            self._run_cycle_inner(is_4h_candle=True)
        finally:
            self._write_ai_log()

    def _layer2_position_management(self) -> None:
        """Layer 2: Position monitoring (every 5 min).

        - Trailing stop updates
        - Close condition evaluation
        """
        try:
            self.broker.ensure_session()
            open_positions = self.broker.get_open_positions()
            self._monitor_positions(open_positions)

            # Update portfolio state
            account = self.broker.get_account_info()
            if account:
                self.portfolio.update_from_broker(account)
        except Exception as e:
            logger.error(f"[Layer2] Error: {e}")

    def _daily_reset(self) -> None:
        """Daily reset callback: summary, Gemini budget, etc."""
        today = datetime.now().strftime("%Y-%m-%d")
        yesterday = self.state.last_summary_date
        if yesterday and yesterday != today:
            self._send_daily_summary(yesterday)
        self.state.last_summary_date = today

    def _send_daily_summary(self, date: str) -> None:
        """Send daily P&L summary via LINE notification."""
        try:
            history = self.position_store.get_trade_history(limit=100)
            day_trades = [t for t in history if (t.get("closed_at") or "").startswith(date)]
            daily_pnl = sum(t.get("pnl") or 0 for t in day_trades)
            wins = [t for t in day_trades if (t.get("pnl") or 0) > 0]
            losses = [t for t in day_trades if (t.get("pnl") or 0) < 0]
            win_rate = len(wins) / len(day_trades) * 100 if day_trades else 0

            pnl_sign = "+" if daily_pnl >= 0 else ""
            emoji = "✅" if daily_pnl >= 0 else "❌"

            lines = [
                f"{emoji} デイリーサマリー ({date})",
                f"損益: {pnl_sign}¥{daily_pnl:,.0f}",
                f"取引数: {len(day_trades)}件 (勝: {len(wins)} / 負: {len(losses)})",
                f"勝率: {win_rate:.0f}%",
            ]
            if day_trades:
                best = max(day_trades, key=lambda t: t.get("pnl") or 0)
                worst = min(day_trades, key=lambda t: t.get("pnl") or 0)
                lines.append(
                    f"最良: {best.get('pair_name', '')} {best.get('pnl', 0):+,.0f}円"
                )
                lines.append(
                    f"最悪: {worst.get('pair_name', '')} {worst.get('pnl', 0):+,.0f}円"
                )

            open_pos = self.position_store.get_open_positions()
            lines.append(f"保有中ポジション: {len(open_pos)}件")

            # Groqレジーム情報を追加
            regime = self.risk_manager.adaptive.market_regime
            if regime and getattr(regime, "regime", None):
                lines.append(f"\n📊 現在レジーム: {regime.regime}")
                if getattr(regime, "reasoning", ""):
                    lines.append(f"Groq判断: {regime.reasoning[:80]}")

            # 翌日の高インパクト経済指標を追加（Groqニュース）
            calendar = self._last_sentiment.get("economic_calendar", [])
            high_impact = [e for e in calendar if str(e.get("impact", "")).upper() in ("HIGH", "★★★", "3")]
            if high_impact:
                lines.append("\n⚠️ 注目経済指標（今後48h）:")
                for ev in high_impact[:4]:
                    lines.append(
                        f"  {ev.get('date', '')} {ev.get('time', '')} "
                        f"[{ev.get('currency', '')}] {ev.get('event', '')}"
                    )
            elif calendar:
                # 高インパクトがなければ直近3件のみ
                lines.append("\n📅 経済指標（今後48h）:")
                for ev in calendar[:3]:
                    lines.append(
                        f"  {ev.get('date', '')} {ev.get('time', '')} "
                        f"[{ev.get('currency', '')}] {ev.get('event', '')}"
                    )

            message = "\n".join(lines)
            logger.info(f"Sending daily summary for {date}")
            self.notifier.send("デイリーサマリー", message)
        except Exception as e:
            logger.error(f"Failed to send daily summary: {e}")

    def _weekly_threshold_review(self) -> None:
        """週次戦略レビュー。

        フロー:
        1. 直近20トレードのPFを計算（ルールベース）
        2. PFが境界値を超えた場合、Groq AIに変更の妥当性を問い合わせ
        3. AIが承認した場合のみ閾値を変更
        4. 結果をLINEで通知
        """
        try:
            rm = self.config.risk_management
            step = int(rm.get("confidence_threshold_step", 5))
            current = self.config.confidence_threshold

            history = self.position_store.get_trade_history(limit=20)
            if len(history) < 10:
                logger.info("[週次レビュー] トレード数が少ないため閾値調整をスキップ")
                return

            pnls = [t.get("pnl") or 0.0 for t in history]
            gross_profit = sum(p for p in pnls if p > 0)
            gross_loss = abs(sum(p for p in pnls if p < 0))
            pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

            # ルールによる提案
            if pf >= 1.3:
                proposed = current + step
                rule_reason = f"PF={pf:.2f} >= 1.3 → 引き上げ提案"
            elif pf < 1.0:
                proposed = current - step
                rule_reason = f"PF={pf:.2f} < 1.0 → 引き下げ提案"
            else:
                logger.info(
                    f"[週次レビュー] PF={pf:.2f} (1.0〜1.3) → 閾値 {current}% 維持"
                )
                return

            # 現在の相場レジームを取得（Geminiが設定している値）
            regime_info = "不明"
            try:
                ar = self.risk_manager.adaptive
                if hasattr(ar, "market_regime") and ar.market_regime:
                    regime_info = (
                        f"{ar.market_regime.regime} / "
                        f"ボラ:{ar.market_regime.volatility_level} / "
                        f"トレンド強度:{ar.market_regime.trend_strength}"
                    )
            except Exception:
                pass

            # Groq AIに変更の妥当性を確認
            logger.info(f"[週次レビュー] Groq AIに判断を依頼中... ({rule_reason})")
            review = self.groq_reviewer.review_weekly_performance(
                trades=history,
                current_threshold=current,
                proposed_threshold=proposed,
                market_regime=regime_info,
            )

            approved = review.get("approved", True)
            final_thr = review.get("final_threshold", proposed)
            ai_reason = review.get("reasoning", "")
            warnings = review.get("warnings", [])

            if approved:
                self.config.confidence_threshold = final_thr
                actual = self.config.confidence_threshold
                logger.info(
                    f"[週次レビュー] AI承認: {current}% → {actual}% | {ai_reason}"
                )
                change_label = f"{current}% → {actual}%"
                status = "✅ 変更実行"
            else:
                actual = current
                logger.info(
                    f"[週次レビュー] AI却下: 閾値 {current}% 維持 | {ai_reason}"
                )
                change_label = f"{current}% 維持"
                status = "🚫 変更見送り"

            warn_text = "\n".join(f"⚠️ {w}" for w in warnings) if warnings else ""
            msg = (
                f"🤖 週次戦略レビュー\n"
                f"直近{len(history)}トレード PF: {pf:.2f}\n"
                f"ルール判断: {rule_reason}\n"
                f"AI判断: {status}\n"
                f"信頼度閾値: {change_label}\n"
                f"AI理由: {ai_reason}"
            )
            if warn_text:
                msg += f"\n{warn_text}"

            self.notifier.send("週次戦略レビュー", msg, line=False)

        except Exception as e:
            logger.error(f"Weekly threshold review failed: {e}")

    def _notify_analysis_result(
        self,
        instrument: str,
        pair_name: str,
        signals: list,
        mtf_data: dict,
    ) -> None:
        """Send Gemini analysis result to LINE after every analysis cycle.

        Called whenever signals are detected, regardless of trade execution.
        Sends chart image + AI reasoning so the user can see what was analyzed.
        """
        primary_df = mtf_data.get("primary")
        decimals = self.broker._price_precision(instrument) if hasattr(self.broker, '_price_precision') else (3 if "JPY" in instrument else 5)

        best = signals[0]
        direction_emoji = "📈" if best.direction.value == "BUY" else "📉"
        thr = self.config.confidence_threshold

        if best.confidence >= thr:
            status = "✅ トレード対象"
        else:
            status = f"⚠️ 信頼度不足 ({best.confidence:.0f}% < {thr}%)"

        current_price = float(primary_df["close"].iloc[-1]) if primary_df is not None and not primary_df.empty else 0

        title = f"{direction_emoji} 分析結果: {pair_name}"
        message = (
            f"現在値: {current_price:.{decimals}f}\n"
            f"方向: {best.direction.value}\n"
            f"パターン: {best.pattern.value}\n"
            f"信頼度: {best.confidence:.0f}%\n"
            f"R:R: {best.risk_reward_ratio:.1f}\n"
            f"ステータス: {status}"
        )
        fields = {
            "Entry": f"{best.entry_price:.{decimals}f}",
            "SL": f"{best.stop_loss:.{decimals}f}",
            "TP": f"{best.take_profit:.{decimals}f}",
            "MTF確認": "✓" if best.mtf_confirmed else "✗",
            "AI確認": "✓" if best.ai_confirmed else "✗",
        }

        chart_image = None
        if self.ai_analyzer and primary_df is not None:
            chart_image = self.ai_analyzer.generate_chart_image(
                primary_df, pair_name,
                self.config.timeframes.get("primary", "HOUR_4"),
            )

        gemini_reasoning = best.reasoning or ""
        self.notifier.send_with_chart(
            title=title,
            message=message,
            fields=fields,
            chart_image=chart_image,
            gemini_response=gemini_reasoning,
        )

    def run_analysis_notify(self) -> None:
        """Run analysis for all active pairs and send results to LINE immediately.

        Used by --notify-analysis CLI flag for on-demand testing.
        """
        if not self.broker.connect():
            logger.error("Failed to connect to broker")
            return

        account = self.broker.get_account_info()
        balance = account.get("balance", 0) if account else 0

        for instrument in self.config.active_pairs:
            try:
                pair_config = self.config.pair_configs.get(instrument, {})
                pair_name = pair_config.get("name", instrument)
                logger.info(f"[notify-analysis] Analyzing {pair_name}...")

                timeframe_config = self.config.timeframes
                mtf_data = {}
                for tf_name in ["primary", "secondary", "trend"]:
                    resolution = timeframe_config.get(tf_name)
                    if resolution:
                        df = self.data_manager.fetch_prices(instrument, resolution, num_points=500)
                        if not df.empty:
                            df = self.tech_analyzer.add_all_indicators(df)
                            mtf_data[tf_name] = df

                if "primary" not in mtf_data:
                    logger.warning(f"No data for {pair_name}")
                    continue

                signals = self.strategy_engine.analyze(
                    epic=instrument,
                    pair_name=pair_name,
                    mtf_data=mtf_data,
                    timeframe_config=timeframe_config,
                )

                if signals:
                    self._notify_analysis_result(instrument, pair_name, signals, mtf_data)
                    logger.info(f"[notify-analysis] Sent {len(signals)} signal(s) for {pair_name}")
                else:
                    logger.info(f"[notify-analysis] No signals for {pair_name}")

            except Exception as e:
                logger.error(f"[notify-analysis] Error for {instrument}: {e}")

        self.broker.disconnect()

    def run_trade_once(self) -> None:
        """Run one trading cycle immediately (analyze + execute trades if signals found).

        Used by --trade-once CLI flag for on-demand trade execution.
        """
        if not self.broker.connect():
            logger.error("Failed to connect to broker")
            return

        account = self.broker.get_account_info()
        if account:
            logger.info(
                f"Account: {account.get('account_name', 'N/A')} | "
                f"Balance: ¥{account.get('balance', 0):,.0f}"
            )
            self.risk_manager.state.peak_balance = account.get("balance", 0)

        logger.info("[trade-once] Running single trading cycle...")
        self._trading_cycle()

        self.broker.disconnect()
        logger.info("[trade-once] Done")

    def stop(self) -> None:
        """Stop the trading bot and disconnect."""
        self._running = False
        self.broker.disconnect()
        logger.info("Trading bot stopped")

    @staticmethod
    def _is_market_closed(now: datetime | None = None) -> tuple[bool, str]:
        """Check if the market should be treated as closed.

        Returns (True, reason) when:
        - Saturday or Sunday (FX market closed globally)
        - Japanese public holiday (per user request)

        Args:
            now: Override current time (for testing). Defaults to datetime.now().

        Returns:
            Tuple of (is_closed, reason).
        """
        now = now or datetime.now()
        weekday = now.weekday()  # 0=Mon, 6=Sun
        if weekday == 5:
            return True, "Saturday (FX market closed)"
        if weekday == 6:
            return True, "Sunday (FX market closed)"
        if _JPHOLIDAY_AVAILABLE and jpholiday.is_holiday(now.date()):
            holiday_name = jpholiday.is_holiday_name(now.date()) or "Japanese holiday"
            return True, f"Japanese holiday ({holiday_name})"
        return False, ""

    def _trading_cycle(self) -> None:
        """Execute one complete trading cycle (legacy entry point).

        Used by run_trade_once and backcompat. Delegates to Layer 0/1 via
        4H candle detection.
        """
        closed, reason = self._is_market_closed()
        if closed:
            logger.info(f"Market closed: {reason} — skipping cycle")
            return

        logger.info("-" * 40)
        logger.info(f"Trading cycle: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        is_4h = self.scheduler.is_new_4h_candle()
        if is_4h:
            self.gemini_budget.start_new_cycle(self.scheduler.current_4h_slot())

        try:
            self._run_cycle_inner(is_4h_candle=is_4h)
        finally:
            self._write_ai_log()

    def _run_cycle_inner(self, *, is_4h_candle: bool = False) -> None:
        """Core cycle logic. _write_ai_log is guaranteed by caller."""
        self.state.last_cycle_started_at = datetime.now().isoformat()

        # Ensure session is alive
        self.broker.ensure_session()

        # Get account state
        account = self.broker.get_account_info()
        balance = account.get("balance", 0)

        # Update portfolio service
        if account:
            self.portfolio.update_from_broker(account)

        # Check if trading is allowed (analysis always runs; only entry is blocked)
        can_trade, reason = self.risk_manager.can_trade(balance)
        if not can_trade:
            logger.info(f"New entries paused: {reason}")

        # Adaptive: check for forced cooldown after consecutive losses
        cooldown, cooldown_reason = self.risk_manager.adaptive.should_cooldown()
        if cooldown:
            logger.info(f"Adaptive cooldown: {cooldown_reason}")

        # Update open position count
        open_positions = self.broker.get_open_positions()
        if not open_positions.empty:
            self.risk_manager.update_open_positions(len(open_positions))
        else:
            self.risk_manager.update_open_positions(0)

        # Monitor existing positions (trailing stop, etc.)
        self._monitor_positions(open_positions)

        # AI Market Regime Analysis (once per cycle, using first active pair)
        self._update_market_regime()

        if is_4h_candle:
            logger.info(
                f"[4H Candle] New 4H candle closed — Gemini analysis eligible "
                f"(budget: {self._gemini_daily_count}/{self._gemini_daily_limit})"
            )
        else:
            logger.debug(f"[4H Candle] Quick mode — next Gemini at {self._next_4h_time} JST")

        # Analyze each active pair
        for instrument in self.config.active_pairs:
            try:
                self._analyze_and_trade(instrument, balance, is_4h_candle=is_4h_candle)
            except Exception as e:
                logger.error(f"Error analyzing {instrument}: {e}")
                logger.debug(traceback.format_exc())

        # Log risk summary with adaptive info
        risk_summary = self.risk_manager.get_risk_summary()
        self.state.last_cycle_finished_at = datetime.now().isoformat()
        adaptive = risk_summary.get("adaptive", {})
        logger.info(
            f"Risk: daily_pnl=¥{risk_summary['daily_pnl']:+,.0f} | "
            f"drawdown={risk_summary['drawdown_pct']:.1f}% | "
            f"trades={risk_summary['daily_trades']} | "
            f"positions={risk_summary['open_positions']} | "
            f"regime={adaptive.get('market_regime', 'N/A')} | "
            f"win_rate={adaptive.get('recent_win_rate', 'N/A')}"
        )

        # ドローダウン警告通知（3%・5%・7%の閾値を超えたら1回だけ送信）
        dd_pct = risk_summary["drawdown_pct"]
        for dd_thr in (3.0, 5.0, 7.0):
            if dd_pct >= dd_thr and dd_thr not in self._dd_warned_levels:
                self._dd_warned_levels.add(dd_thr)
                self.notifier.drawdown_warning(
                    drawdown_pct=dd_pct,
                    threshold_pct=dd_thr,
                    balance=account.get("balance", 0),
                )
            elif dd_pct < dd_thr:
                # 回復したらリセット（次回また警告できるように）
                self._dd_warned_levels.discard(dd_thr)

        # Send daily summary when the date rolls over (triggers once per day)
        today = datetime.now().strftime("%Y-%m-%d")
        if self._last_summary_date and self._last_summary_date != today:
            self._send_daily_summary(self._last_summary_date)
        self._last_summary_date = today

        # 週次閾値自動調整（毎週月曜に実行）
        this_week = datetime.now().strftime("%Y-W%W")
        if self._last_weekly_review_date != this_week and datetime.now().weekday() == 0:
            self._weekly_threshold_review()
            self._last_weekly_review_date = this_week

    def _write_ai_log(self) -> None:
        """Write per-cycle AI analysis results to data/ai_analysis_log.json for dashboard."""
        try:
            import json as _json
            ar = self.risk_manager.adaptive
            regime_data: dict[str, Any] = {}
            if hasattr(ar, "market_regime") and ar.market_regime:
                mr = ar.market_regime
                regime_data = {
                    "regime": getattr(mr, "regime", ""),
                    "volatility_level": getattr(mr, "volatility_level", ""),
                    "trend_strength": getattr(mr, "trend_strength", ""),
                    "reasoning": getattr(mr, "reasoning", ""),
                    "risk_multiplier": getattr(mr, "risk_multiplier", 1.0),
                }

            import os as _os

            gemini_used = int(self._gemini_daily_count)
            gemini_limit = int(self._gemini_daily_limit)
            gemini_remaining = max(0, gemini_limit - gemini_used)

            def _safe_int(value: Any) -> int | None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    return None

            groq_used = _safe_int(getattr(self.groq_reviewer, "daily_count", None))
            groq_limit = _safe_int(getattr(self.groq_reviewer, "daily_limit", None))
            groq_remaining = (
                max(0, groq_limit - groq_used)
                if groq_used is not None and groq_limit is not None
                else None
            )

            payload = {
                "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "pid": _os.getpid(),
                "regime": regime_data,
                "pairs": self._cycle_log,
                "budget": {
                    "gemini": {
                        "daily_used": gemini_used,
                        "daily_hard_limit": gemini_limit,
                        "daily_remaining": gemini_remaining,
                    },
                    "groq": {
                        "available": bool(getattr(self.groq_reviewer, "available", False)),
                        "daily_used": groq_used,
                        "daily_hard_limit": groq_limit,
                        "daily_remaining": groq_remaining,
                    },
                },
            }
            log_path = Path("data/ai_analysis_log.json")
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "w", encoding="utf-8") as f:
                _json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.debug(f"_write_ai_log failed: {e}")

    def _update_market_regime(self) -> None:
        """Groq + 外部センチメントで相場レジームを判断し、リスク管理に反映する。

        Groq TPD節約のため1時間に1回のみ実行。
        """
        import time as _time

        # 1時間キャッシュ: 前回のレジーム分析から1時間以内ならスキップ
        now_ts = _time.time()
        if not hasattr(self, "_last_regime_ts"):
            self._last_regime_ts = 0.0
        if now_ts - self._last_regime_ts < 3600:
            return
        self._last_regime_ts = now_ts

        try:
            instrument = self.config.active_pairs[0]
            pair_config = self.config.pair_configs.get(instrument, {})
            pair_name = pair_config.get("name", instrument)
            resolution = self.config.timeframes.get("primary", "HOUR_4")
            df = self.data_manager.fetch_prices(instrument, resolution, num_points=200)

            if df.empty:
                return

            df = self.tech_analyzer.add_all_indicators(df)
            last = df.iloc[-1]

            # テクニカル指標を辞書化
            indicators = {
                "atr": round(float(last.get("atr", 0)), 5),
                "adx": round(float(last.get("adx", 0)), 1),
                "rsi": round(float(last.get("rsi", 50)), 1),
                "price": round(float(last.get("close", 0)), 5),
                "ma20": round(float(last.get("sma_20", 0)), 5),
                "ma200": round(float(last.get("sma_200", 0)), 5),
                "bb_width": round(
                    float(last.get("bb_upper", 0)) - float(last.get("bb_lower", 0)), 5
                ),
            }

            # センチメントデータを1時間キャッシュ
            now_ts = _time.time()
            if now_ts - self._last_sentiment_ts > 3600:
                self._last_sentiment = self.sentiment_fetcher.fetch_all(
                    self.config.active_pairs
                )
                self._last_sentiment_ts = now_ts

            # Groqで相場レジームを判断
            regime_data = self.groq_reviewer.analyze_market_regime(
                indicators=indicators,
                sentiment_data=self._last_sentiment,
                pair_name=pair_name,
            )

            new_regime = regime_data.get("regime", "ranging")
            from src.models.risk_manager import MarketRegime
            self.risk_manager.adaptive.market_regime = MarketRegime(
                regime=new_regime,
                volatility_level=regime_data.get("volatility_level", "medium"),
                trend_strength=regime_data.get("trend_strength", "moderate"),
                risk_multiplier=regime_data.get("risk_multiplier", 1.0),
                confidence=regime_data.get("confidence", 50),
                reasoning=regime_data.get("reasoning", ""),
            )

            # レジーム変化通知 + 保有ポジションのSL/TP調整
            if self._prev_regime and self._prev_regime != new_regime:
                self.notifier.regime_changed(
                    pair=pair_name,
                    old_regime=self._prev_regime,
                    new_regime=new_regime,
                    volatility=regime_data.get("volatility_level", "medium"),
                    trend=regime_data.get("trend_strength", "moderate"),
                    risk_multiplier=regime_data.get("risk_multiplier", 1.0),
                    reasoning=regime_data.get("reasoning", ""),
                )
                # 保有中ポジションのSL/TPをレジームに合わせて再計算
                self._adjust_open_positions_for_regime(new_regime)
            self._prev_regime = new_regime

        except Exception as e:
            logger.error(f"Market regime update failed: {e}")

    def _analyze_and_trade(
        self, instrument: str, balance: float, *, is_4h_candle: bool = False,
    ) -> None:
        """Analyze a single currency pair and execute trades if conditions met.

        Args:
            instrument: OANDA instrument (e.g., 'USD_JPY').
            balance: Current account balance.
            is_4h_candle: True if a new 4H candle just closed (Gemini eligible).
        """
        pair_config = self.config.pair_configs.get(instrument, {})
        pair_name = pair_config.get("name", instrument)
        pip_value = self.broker.get_pip_value(instrument) if hasattr(self.broker, 'get_pip_value') else pair_config.get("pip_value", 0.01)

        logger.info(f"Analyzing {pair_name} ({instrument})...")

        # Fetch multi-timeframe data (分析は常に実行 — ポジション上限はエントリー時にチェック)
        timeframe_config = self.config.timeframes
        mtf_data = {}
        for tf_name in ["primary", "secondary", "trend"]:
            resolution = timeframe_config.get(tf_name)
            if resolution:
                df = self.data_manager.fetch_prices(instrument, resolution, num_points=500)
                if not df.empty:
                    # Add indicators
                    df = self.tech_analyzer.add_all_indicators(df)
                    mtf_data[tf_name] = df

        if "primary" not in mtf_data:
            logger.warning(f"No primary data for {pair_name}")
            return

        # 2. Groq directional vote for consensus (Layer 3)
        primary_df = mtf_data["primary"]
        last = primary_df.iloc[-1]
        indicators_for_vote = {
            "atr": round(float(last.get("atr", 0)), 5),
            "adx": round(float(last.get("adx", 0)), 1),
            "rsi": round(float(last.get("rsi", 50)), 1),
            "price": round(float(last.get("close", 0)), 5),
            "ma20": round(float(last.get("sma_20", 0)), 5),
            "ma200": round(float(last.get("sma_200", 0)), 5),
        }
        # Groq方向性投票: 4H確定時のみ実行、それ以外はキャッシュ使用（TPD節約）
        if not hasattr(self, "_groq_vote_cache"):
            self._groq_vote_cache: dict[str, dict | None] = {}

        if is_4h_candle and self.groq_reviewer.available:
            groq_vote = self.groq_reviewer.get_directional_vote(
                pair_name=pair_name,
                indicators=indicators_for_vote,
                sentiment_data=self._last_sentiment,
                current_regime=self.risk_manager.adaptive.market_regime.regime,
            )
            self._groq_vote_cache[instrument] = groq_vote
        else:
            groq_vote = self._groq_vote_cache.get(instrument)

        # 3. Run strategy engine (Phase 1: quick mode without Gemini)
        #    First pass always runs without Gemini to get algorithm signals
        signals_quick = self.strategy_engine.analyze(
            epic=instrument,
            pair_name=pair_name,
            mtf_data=mtf_data,
            timeframe_config=timeframe_config,
            groq_vote=groq_vote,
            sentiment_data=self._last_sentiment,
            use_gemini=False,
        )

        # 3b. Gemini: 4H足確定時 + 事前フィルター通過 + 予算内のみ実行
        use_gemini_this_cycle = False
        gemini_skip_reason = ""
        if not is_4h_candle:
            gemini_skip_reason = f"次回 {self._next_4h_time} 確定時に分析"
        elif not self._gemini_budget_ok():
            gemini_skip_reason = f"本日の上限到達 ({self._gemini_daily_count}/{self._gemini_daily_limit})"
        elif not self._should_use_gemini(instrument, signals_quick):
            gemini_skip_reason = "対象外（パターン未検出）"
        else:
            use_gemini_this_cycle = True

        signals = signals_quick
        if use_gemini_this_cycle:
            logger.info(f"[Gemini 4H] {pair_name}: Gemini分析を実行")
            signals = self.strategy_engine.analyze(
                epic=instrument,
                pair_name=pair_name,
                mtf_data=mtf_data,
                timeframe_config=timeframe_config,
                groq_vote=groq_vote,
                sentiment_data=self._last_sentiment,
                use_gemini=True,
            )
            self._use_gemini_budget()

            # Gemini結果をキャッシュ（次回5分サイクル用）
            _ai_sig = self.strategy_engine.last_ai_signal
            if _ai_sig and _ai_sig.is_valid:
                self._gemini_cache[instrument] = {
                    "direction": _ai_sig.direction.value,
                    "confidence": round(_ai_sig.confidence, 1),
                    "reasoning": (_ai_sig.reasoning[:120] if _ai_sig.reasoning else ""),
                    "analyzed_at": datetime.now().strftime("%H:%M"),
                }
        else:
            logger.debug(f"[Gemini] {pair_name}: スキップ — {gemini_skip_reason}")

        # Gemini投票情報を構築（実行時 or キャッシュ or スキップ）
        _ai_sig = self.strategy_engine.last_ai_signal if use_gemini_this_cycle else None
        cached_gemini = self._gemini_cache.get(instrument)
        if _ai_sig and _ai_sig.is_valid:
            gemini_vote = {
                "direction": _ai_sig.direction.value,
                "confidence": round(_ai_sig.confidence, 1),
                "reasoning": (_ai_sig.reasoning[:120] if _ai_sig.reasoning else ""),
            }
        elif cached_gemini:
            gemini_vote = dict(cached_gemini)  # use cached result
            gemini_skip_reason = f"キャッシュ ({cached_gemini.get('analyzed_at', '')}時点)"
        else:
            gemini_vote = {
                "direction": "N/A",
                "confidence": 0,
                "reasoning": "",
            }

        # --- 総合判断（3票多数決）を集計 ---
        groq_dir_str = "NONE"
        groq_conf_val = 0
        if groq_vote:
            gd = groq_vote.get("direction", "NONE")
            groq_dir_str = gd.value if hasattr(gd, "value") else str(gd)
            groq_conf_val = groq_vote.get("confidence", 0)

        gemini_dir_str = gemini_vote.get("direction", "N/A")
        gemini_conf_val = gemini_vote.get("confidence", 0)

        # Algorithm vote = best signal direction (if any)
        algo_dir_str = signals[0].direction.value if signals else "NONE"
        algo_conf_val = round(signals[0].confidence, 1) if signals else 0

        # Count votes per direction
        vote_list = [
            {"source": "Algorithm", "direction": algo_dir_str, "confidence": algo_conf_val},
            {"source": "Gemini", "direction": gemini_dir_str, "confidence": gemini_conf_val},
            {"source": "Groq", "direction": groq_dir_str, "confidence": groq_conf_val},
        ]
        dir_counts: dict[str, int] = {}
        dir_confs: dict[str, list[float]] = {}
        for v in vote_list:
            d = v["direction"]
            if d in ("NONE", "N/A", "NEUTRAL"):
                continue
            dir_counts[d] = dir_counts.get(d, 0) + 1
            dir_confs.setdefault(d, []).append(v["confidence"])

        # Determine overall verdict
        overall_direction = "NONE"
        overall_confidence = 0
        overall_agree = 0
        if dir_counts:
            best_dir = max(dir_counts, key=lambda k: dir_counts[k])
            overall_agree = dir_counts[best_dir]
            if overall_agree >= 2:
                overall_direction = best_dir
                overall_confidence = round(sum(dir_confs[best_dir]) / len(dir_confs[best_dir]), 1)

        # サイクルログに分析結果を記録（シグナルなし含む）
        self._cycle_log[instrument] = {
            "pair_name": pair_name,
            "analyzed_at": datetime.now().strftime("%H:%M:%S"),
            "overall": {
                "direction": overall_direction,
                "confidence": overall_confidence,
                "agree_count": overall_agree,
                "total_votes": 3,
            },
            "votes": vote_list,
            "groq_vote": {
                "direction": groq_dir_str,
                "confidence": groq_conf_val,
                "reasoning": groq_vote.get("reasoning", "") if groq_vote else "",
            },
            "gemini_vote": gemini_vote,
            "gemini_status": {
                "active": use_gemini_this_cycle,
                "skip_reason": gemini_skip_reason,
                "next_4h_time": self._next_4h_time,
                "budget_used": self._gemini_daily_count,
                "budget_limit": self._gemini_daily_limit,
                "cached": bool(cached_gemini) and not use_gemini_this_cycle,
            },
            "signals": [
                {
                    "direction": s.direction.value,
                    "pattern": s.pattern.value,
                    "confidence": round(s.confidence, 1),
                    "consensus_confirmed": getattr(s, "consensus_confirmed", False),
                    "ai_confirmed": s.ai_confirmed,
                    "mtf_confirmed": s.mtf_confirmed,
                    "rr_ratio": round(s.risk_reward_ratio, 2),
                    "reasoning": s.reasoning[:120] if s.reasoning else "",
                    "sources": s.sources,
                }
                for s in signals
            ] if signals else [],
        }

        if not signals:
            logger.info(f"No signals for {pair_name}")
            return

        # 2b. Send Gemini analysis result to LINE (regardless of trade execution)
        self._notify_analysis_result(instrument, pair_name, signals, mtf_data)

        # 2c. Entry gate: エントリーは4H足確定時のみ
        if not is_4h_candle:
            logger.info(
                f"[Entry gate] {pair_name}: 分析完了・エントリーは次回4H確定時 "
                f"({self._next_4h_time} JST)"
            )
            return

        # 2d. Position limit check (分析後にチェック — 分析自体は常に実行)
        max_per_pair = self.config.risk_management.get("max_positions_per_pair", 1)
        existing = self._open_trades.get(instrument, [])
        if len(existing) >= max_per_pair:
            logger.info(
                f"Max positions per pair reached for {pair_name} "
                f"({len(existing)}/{max_per_pair}), entry blocked"
            )
            return

        # 3. Take the best signal (with consensus + adaptive pattern filtering + confidence threshold)
        thr = self.config.confidence_threshold
        best_signal = None
        for sig in signals:
            # Consensus gate: require 2/3 AI agreement to enter
            if not getattr(sig, "consensus_confirmed", False):
                logger.info(
                    f"Signal skipped for {pair_name}: consensus not reached "
                    f"({sig.pattern.value} {sig.direction.value})"
                )
                continue
            if sig.confidence < thr:
                logger.debug(
                    f"Signal confidence {sig.confidence}% < threshold {thr}%: skip"
                )
                continue
            skip, skip_reason = self.risk_manager.adaptive.should_skip_pattern(
                sig.pattern.value
            )
            if skip:
                logger.info(f"Adaptive skip for {pair_name}: {skip_reason}")
                continue
            best_signal = sig
            break

        if best_signal is None:
            logger.info(f"No viable signals for {pair_name} (all filtered)")
            return

        # 4. Validate signal
        valid, reason = self.risk_manager.validate_signal(best_signal)
        if not valid:
            logger.info(f"Signal rejected for {pair_name}: {reason}")
            return

        # Re-check trading allowed
        can_trade, reason = self.risk_manager.can_trade(balance)
        if not can_trade:
            logger.warning(f"Trading not allowed: {reason}")
            return

        # 5. Calculate position size
        # Recalculate SL/TP using ATR from current data
        primary_df = mtf_data["primary"]
        atr = float(primary_df["atr"].iloc[-1]) if "atr" in primary_df.columns else 0

        if atr > 0:
            refined_sl = self.risk_manager.calculate_stop_loss(
                direction=best_signal.direction.value,
                entry_price=best_signal.entry_price,
                atr=atr,
                pattern_sl=best_signal.stop_loss,
                broker=self.broker,
                instrument=instrument,
            )
            refined_tp = self.risk_manager.calculate_take_profit(
                direction=best_signal.direction.value,
                entry_price=best_signal.entry_price,
                stop_loss=refined_sl,
            )
            best_signal.stop_loss = refined_sl
            best_signal.take_profit = refined_tp

        pos_size = self.risk_manager.calculate_position_size(
            signal=best_signal,
            account_balance=balance,
            pip_value=pip_value,
            broker=self.broker,
            instrument=instrument,
        )

        adaptive_mult = self.risk_manager.adaptive.calculate_risk_multiplier(
            self.risk_manager.state.current_drawdown_pct
        )
        logger.info(
            f"Trade setup: {pair_name} {best_signal.direction.value} | "
            f"Entry={best_signal.entry_price:.5f} | "
            f"SL={best_signal.stop_loss:.5f} | "
            f"TP={best_signal.take_profit:.5f} | "
            f"Units={pos_size.units:,} ({pos_size.lots:.4f} std lots) | "
            f"Risk=¥{pos_size.risk_amount:,.0f} ({pos_size.risk_pct:.2f}%) | "
            f"R:R={best_signal.risk_reward_ratio:.1f} | "
            f"Adaptive×{adaptive_mult:.2f} | "
            f"Regime={self.risk_manager.adaptive.market_regime.regime}"
        )

        # 6. Generate chart image for notification
        chart_image = None
        gemini_reasoning = best_signal.reasoning or ""
        if self.ai_analyzer and "primary" in mtf_data:
            chart_image = self.ai_analyzer.generate_chart_image(
                mtf_data["primary"], pair_name,
                self.config.timeframes.get("primary", "HOUR_4"),
            )

        # 7. Final duplicate check before execution
        existing = self._open_trades.get(instrument, [])
        max_per_pair = self.config.risk_management.get("max_positions_per_pair", 1)
        if len(existing) >= max_per_pair:
            logger.warning(f"BLOCKED duplicate trade for {pair_name} at final check")
            return

        # 既存ポジションと逆方向なら拒否
        if existing:
            existing_direction = existing[0].get("direction", "")
            if existing_direction and existing_direction != best_signal.direction.value:
                logger.info(
                    f"Opposite direction rejected for {pair_name}: "
                    f"existing={existing_direction}, new={best_signal.direction.value}"
                )
                return

        # 8. Execute trade (always paper mode)
        self._execute_trade(
            instrument, pair_name, best_signal, pos_size,
            chart_image=chart_image, gemini_reasoning=gemini_reasoning,
        )

    def _execute_trade(
        self,
        instrument: str,
        pair_name: str,
        signal: TradeSignal,
        pos_size: Any,
        chart_image: bytes | None = None,
        gemini_reasoning: str = "",
    ) -> None:
        """Execute a paper trade.

        Args:
            instrument: OANDA instrument.
            pair_name: Currency pair name.
            signal: Trade signal.
            pos_size: Position size calculation.
            chart_image: Chart PNG bytes for LINE notification.
            gemini_reasoning: Gemini AI analysis text.
        """
        result = self.broker.open_position(
            epic=instrument,
            direction=signal.direction.value,
            size=pos_size.units,
            order_type="MARKET",
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            trailing_stop=self.risk_manager.trailing_stop,
        )

        if result.get("success"):
            deal_id = result["deal_id"]
            trade_info = {
                "deal_id": deal_id,
                "instrument": instrument,
                "pair_name": pair_name,
                "direction": signal.direction.value,
                "entry_price": result.get("level", signal.entry_price),
                "stop_loss": signal.stop_loss,
                "take_profit": signal.take_profit,
                "size": pos_size.units,
                "pattern": signal.pattern.value,
                "opened_at": datetime.now().isoformat(),
            }
            if instrument not in self._open_trades:
                self._open_trades[instrument] = []
            self._open_trades[instrument].append(trade_info)
            self.position_store.save_position(trade_info)

            log_trade(
                f"OPEN|{pair_name}|{signal.direction.value}|"
                f"deal={deal_id}|"
                f"entry={signal.entry_price:.5f}|"
                f"sl={signal.stop_loss:.5f}|"
                f"tp={signal.take_profit:.5f}|"
                f"size={pos_size.units}|"
                f"pattern={signal.pattern.value}"
            )

            # Send notification with chart image + Gemini analysis
            rr = abs(signal.take_profit - signal.entry_price) / abs(signal.entry_price - signal.stop_loss) if abs(signal.entry_price - signal.stop_loss) > 0 else 0
            self.notifier.send_with_chart(
                title=f"{signal.direction.value} {pair_name}",
                message=f"Pattern: {signal.pattern.value}",
                fields={
                    "Entry Price": f"{signal.entry_price:.5f}",
                    "Size": f"{pos_size.units:,} units",
                    "Stop Loss": f"{signal.stop_loss:.5f}",
                    "Take Profit": f"{signal.take_profit:.5f}",
                    "R:R": f"1:{rr:.1f}",
                    "Sources": ", ".join(signal.sources),
                },
                chart_image=chart_image,
                gemini_response=gemini_reasoning,
            )
        else:
            logger.error(
                f"Trade execution failed: {result.get('reason', 'unknown')}"
            )

    def _close_trade_info(self, instrument: str, trade_info: dict) -> None:
        """Handle close processing for a single trade_info dict.

        Uses OANDA actual realizedPL when connected to OANDA API.
        Falls back to estimation only in paper mode.
        """
        deal_id = trade_info.get("deal_id", "")
        pnl = 0.0
        exit_price = 0.0

        # OANDA API: fetch actual realizedPL and exit price
        use_oanda = hasattr(self.broker, "_use_oanda_api") and self.broker._use_oanda_api
        if use_oanda and deal_id:
            closed = self.broker.get_closed_trade(deal_id)
            if closed:
                pnl = closed["realized_pl"]
                exit_price = closed.get("exit_price", 0.0)

        # Paper fallback: estimate from current market price
        if exit_price == 0:
            try:
                price_info = self.data_manager.get_latest_price(instrument)
                exit_price = price_info.get("mid", 0)
            except Exception:
                exit_price = 0
        if not use_oanda:
            pnl = self._estimate_closed_pnl_paper_only(trade_info, exit_price)

        self.risk_manager.record_trade_result(
            pnl=pnl,
            pattern=trade_info.get("pattern", ""),
            instrument=instrument,
        )

        pair_name = instrument.replace("_", "/")
        self.position_store.close_position(
            deal_id=deal_id,
            exit_price=exit_price,
            pnl=pnl,
        )

        # ペーパー残高に確定損益を反映
        if hasattr(self.broker, "apply_trade_pnl"):
            self.broker.apply_trade_pnl(pnl)

        log_trade(
            f"CLOSE|{pair_name}|"
            f"deal={deal_id}|"
            f"exit={exit_price:.5f}|"
            f"pnl={pnl:.2f}|"
            f"pattern={trade_info.get('pattern', '')}"
        )

        logger.info(
            f"Position closed: {instrument} "
            f"deal={deal_id or 'N/A'} "
            f"PnL=¥{pnl:+,.0f}"
        )

        # R:R 計算
        entry_p = trade_info.get("entry_price", 0)
        sl_p = trade_info.get("stop_loss", 0)
        tp_p = trade_info.get("take_profit", 0)
        rr = (
            abs(tp_p - entry_p) / abs(entry_p - sl_p)
            if entry_p and sl_p and abs(entry_p - sl_p) > 0
            else 0.0
        )
        # Pips 計算
        if exit_price and entry_p:
            direction_str = trade_info.get("direction", "")
            pip_unit = self.broker.get_pip_value(instrument) if hasattr(self.broker, 'get_pip_value') else (0.01 if "JPY" in instrument else 0.0001)
            raw_diff = (exit_price - entry_p) if direction_str == "BUY" else (entry_p - exit_price)
            pips_val = raw_diff / pip_unit
        else:
            pips_val = 0.0

        # Notify
        self.notifier.trade_closed(
            pair=pair_name,
            direction=trade_info.get("direction", ""),
            pnl=pnl,
            pips=pips_val,
            entry_price=entry_p,
            exit_price=exit_price,
            opened_at=trade_info.get("opened_at", ""),
            pattern=trade_info.get("pattern", ""),
            risk_reward=rr,
        )

    def _monitor_positions(self, positions_df: pd.DataFrame) -> None:
        """Monitor open positions for trailing stops and exits.

        Args:
            positions_df: DataFrame of open positions.
        """
        # ブローカーのアクティブな deal_id セットを取得
        active_deal_ids: set[str] = set()
        if not positions_df.empty and "dealId" in positions_df.columns:
            active_deal_ids = set(positions_df["dealId"].tolist())

        if positions_df.empty:
            # 全ポジションがクローズされた
            for instrument, positions_list in list(self._open_trades.items()):
                for trade_info in positions_list:
                    self._close_trade_info(instrument, trade_info)
                del self._open_trades[instrument]
            return

        # deal_id 単位でクローズ済みポジションを検出
        for instrument, positions_list in list(self._open_trades.items()):
            still_open: list[dict] = []
            for trade_info in positions_list:
                deal_id = trade_info.get("deal_id", "")
                if active_deal_ids and deal_id not in active_deal_ids:
                    # このポジションはクローズされた
                    self._close_trade_info(instrument, trade_info)
                else:
                    still_open.append(trade_info)

            if still_open:
                self._open_trades[instrument] = still_open
            else:
                del self._open_trades[instrument]

        # Update trailing stops for open positions
        for instrument, positions_list in self._open_trades.items():
            for trade_info in positions_list:
                if not self.risk_manager.trailing_stop:
                    continue

                try:
                    # Get current price
                    price_info = self.data_manager.get_latest_price(instrument)
                    current_price = price_info.get("mid", 0)
                    if current_price == 0:
                        continue

                    # Get current ATR
                    df = self.data_manager.fetch_prices(
                        instrument,
                        self.config.timeframes.get("primary", "HOUR_4"),
                        num_points=20,
                    )
                    if df.empty or "atr" not in df.columns:
                        df = self.tech_analyzer.add_atr(df)
                    atr = float(df["atr"].iloc[-1]) if "atr" in df.columns else 0

                    if atr > 0:
                        new_sl = self.risk_manager.calculate_trailing_stop(
                            direction=trade_info["direction"],
                            current_price=current_price,
                            current_sl=trade_info["stop_loss"],
                            atr=atr,
                        )

                        if new_sl != trade_info["stop_loss"]:
                            old_sl = trade_info["stop_loss"]
                            self.broker.update_position(
                                deal_id=trade_info["deal_id"],
                                stop_level=new_sl,
                                epic=instrument,
                            )
                            trade_info["stop_loss"] = new_sl
                            # SQLiteも更新してダッシュボードに反映
                            self.position_store.update_position(
                                deal_id=trade_info["deal_id"],
                                stop_loss=new_sl,
                            )
                            logger.info(
                                f"Trailing stop updated: {instrument} "
                                f"new SL={new_sl:.5f}"
                            )
                            pair_name = instrument.replace("_", "/")
                            self.notifier.sl_updated(
                                pair=pair_name,
                                direction=trade_info["direction"],
                                old_sl=old_sl,
                                new_sl=new_sl,
                                entry=trade_info["entry_price"],
                            )

                except Exception as e:
                    logger.error(f"Error monitoring {instrument}: {e}")

        # Layer 2: ポジション再評価（3サイクルに1回、Groq+アルゴのみ — Gemini不使用）
        self._position_eval_counter += 1
        if self._position_eval_counter % 3 != 0:
            return
        if not self._open_trades:
            return

        from src.strategies.consensus_engine import ConsensusEngine
        consensus_engine = ConsensusEngine(self.config.consensus_config)
        current_regime = self.risk_manager.adaptive.market_regime.regime

        for instrument, positions_list in list(self._open_trades.items()):
            for trade_info in list(positions_list):
                try:
                    resolution = self.config.timeframes.get("primary", "HOUR_4")
                    df = self.data_manager.fetch_prices(instrument, resolution, num_points=200)
                    if df.empty:
                        continue
                    df = self.tech_analyzer.add_all_indicators(df)

                    price_info = self.data_manager.get_latest_price(instrument)
                    current_price = price_info.get("mid", 0)
                    if current_price == 0:
                        continue

                    pos_dir = trade_info.get("direction", "BUY")
                    entry_price = trade_info.get("entry_price", current_price)
                    pair_name = instrument.replace("_", "/")

                    # Groq がペアごとに方向性を判定
                    last = df.iloc[-1]
                    indicators_l2 = {
                        "atr": round(float(last.get("atr", 0)), 5),
                        "adx": round(float(last.get("adx", 0)), 1),
                        "rsi": round(float(last.get("rsi", 50)), 1),
                        "price": round(float(last.get("close", 0)), 5),
                        "ma20": round(float(last.get("sma_20", 0)), 5),
                        "ma200": round(float(last.get("sma_200", 0)), 5),
                    }
                    groq_dir_vote = self.groq_reviewer.get_directional_vote(
                        pair_name=pair_name,
                        indicators=indicators_l2,
                        sentiment_data=self._last_sentiment,
                        current_regime=current_regime,
                    ) if self.groq_reviewer.available else {
                        "direction": SignalDirection.NONE, "confidence": 50.0,
                    }

                    # Geminiキャッシュがあれば使用、なければNONE
                    cached = self._gemini_cache.get(instrument)
                    if cached and cached.get("direction") not in ("N/A", "NONE"):
                        gemini_dir_raw = cached["direction"]
                        gemini_dir = (
                            SignalDirection.BUY if gemini_dir_raw == "BUY"
                            else SignalDirection.SELL if gemini_dir_raw == "SELL"
                            else SignalDirection.NONE
                        )
                        gemini_conf = cached.get("confidence", 50.0)
                    else:
                        gemini_dir = SignalDirection.NONE
                        gemini_conf = 50.0

                    pos_direction = (
                        SignalDirection.BUY if pos_dir == "BUY" else SignalDirection.SELL
                    )
                    consensus = consensus_engine.decide_close(
                        position_direction=pos_direction,
                        gemini_dir=gemini_dir,
                        gemini_conf=gemini_conf,
                        groq_dir=groq_dir_vote.get("direction", SignalDirection.NONE),
                        groq_conf=groq_dir_vote.get("confidence", 50.0),
                    )

                    if consensus.consensus_reached and consensus.direction != pos_direction:
                        logger.info(
                            f"[Layer2 Close] {instrument}: consensus close "
                            f"{consensus.summary} — closing position"
                        )
                        close_result = self.broker.close_position(trade_info["deal_id"])
                        close_success = (
                            close_result.get("success", False)
                            if isinstance(close_result, dict)
                            else close_result is not None
                        )
                        if close_success:
                            # _close_trade_info handles: record_trade_result,
                            # position_store.close_position, apply_trade_pnl, log, notify
                            self._close_trade_info(instrument, trade_info)
                            positions_list.remove(trade_info)
                    else:
                        logger.debug(
                            f"[Layer2 Eval] {instrument}: HOLD — {consensus.summary}"
                        )

                except Exception as e:
                    logger.error(f"[Layer2] Error evaluating {instrument}: {e}")

        # 空リストのペアを削除
        for instrument in list(self._open_trades.keys()):
            if not self._open_trades[instrument]:
                del self._open_trades[instrument]

    def fix_open_positions_sltp(self) -> None:
        """保有中の全ポジションのSL/TPを修正済みロジックで即時再計算・更新する。

        `python main.py --fix-sltp` から呼び出す。
        レジーム変化を待たずに今すぐ適用したい場合に使用。
        """
        if not self._open_trades or sum(len(v) for v in self._open_trades.values()) == 0:
            logger.info("[fix-sltp] 保有ポジションなし")
            return

        # 現在のレジームを最新状態に更新してから適用
        self._update_market_regime()
        current_regime = self.risk_manager.adaptive.market_regime.regime
        total = sum(len(v) for v in self._open_trades.values())
        logger.info(
            f"[fix-sltp] 現在レジーム: {current_regime} | "
            f"{total}件のポジションを修正します"
        )
        self._adjust_open_positions_for_regime(current_regime)

    def _adjust_open_positions_for_regime(self, new_regime: str) -> None:
        """レジーム変化時に保有ポジションのSL/TPをレジームに合わせて再計算・更新する。

        SLは含み益を守る方向のみ移動（不利方向には動かさない）。
        TPは新レジームの期待値に合わせて再設定する。

        Args:
            new_regime: Groqが判断した新しい相場レジーム名。
        """
        total = sum(len(v) for v in self._open_trades.values())
        if total == 0:
            return

        logger.info(
            f"[Regime調整] レジーム変化({new_regime})により"
            f"{total}件の保有ポジションを見直します"
        )

        for instrument, positions_list in self._open_trades.items():
            for trade_info in positions_list:
                try:
                    direction = trade_info.get("direction", "")
                    entry_price = trade_info.get("entry_price", 0.0)
                    current_sl = trade_info.get("stop_loss", 0.0)
                    deal_id = trade_info.get("deal_id", "")

                    if not entry_price or not deal_id:
                        continue

                    # 現在のATRを取得
                    df = self.data_manager.fetch_prices(
                        instrument,
                        self.config.timeframes.get("primary", "HOUR_4"),
                        num_points=20,
                    )
                    if df.empty:
                        continue
                    if "atr" not in df.columns:
                        df = self.tech_analyzer.add_atr(df)
                    atr = float(df["atr"].iloc[-1]) if "atr" in df.columns else 0
                    if atr == 0:
                        continue

                    # レジーム調整済みSL/TPを計算（min/max pip bounds 適用済み）
                    new_sl = self.risk_manager.calculate_stop_loss(
                        direction=direction,
                        entry_price=entry_price,
                        atr=atr,
                        broker=self.broker,
                        instrument=instrument,
                    )
                    new_tp = self.risk_manager.calculate_take_profit(
                        direction=direction,
                        entry_price=entry_price,
                        stop_loss=new_sl,
                    )

                    # SLは含み益を守る方向のみ移動（不利方向には動かさない）
                    sl_improved = (
                        (direction == "BUY"  and new_sl > current_sl) or
                        (direction == "SELL" and new_sl < current_sl)
                    )
                    sl_to_apply = new_sl if sl_improved else current_sl

                    # ブローカーに反映
                    self.broker.update_position(
                        deal_id=deal_id,
                        stop_level=sl_to_apply,
                        limit_level=new_tp,
                        epic=instrument,
                    )

                    # SQLiteに反映
                    self.position_store.update_position(
                        deal_id=deal_id,
                        stop_loss=sl_to_apply,
                        take_profit=new_tp,
                    )

                    pair_name = instrument.replace("_", "/")
                    decimals = self.broker._price_precision(instrument) if hasattr(self.broker, '_price_precision') else (3 if "JPY" in instrument else 5)

                    logger.info(
                        f"[Regime調整] {pair_name} {direction} | "
                        f"SL: {current_sl:.{decimals}f} → {sl_to_apply:.{decimals}f} | "
                        f"TP: {trade_info.get('take_profit', 0):.{decimals}f} → {new_tp:.{decimals}f}"
                    )

                    # trade_info を更新
                    trade_info["stop_loss"] = sl_to_apply
                    trade_info["take_profit"] = new_tp

                    # SL変化があればLINE通知
                    if sl_to_apply != current_sl:
                        self.notifier.sl_updated(
                            pair=pair_name,
                            direction=direction,
                            old_sl=current_sl,
                            new_sl=sl_to_apply,
                            entry=entry_price,
                        )

                except Exception as e:
                    logger.error(
                        f"[Regime調整] {instrument} の調整に失敗: {e}"
                    )

    def _estimate_closed_pnl_paper_only(
        self, trade_info: dict[str, Any], exit_price: float
    ) -> float:
        """Estimate P&L for a closed position in paper trading mode.

        PAPER MODE ONLY. When connected to OANDA, use get_closed_trade()
        instead to get the actual realizedPL from the API.

        P&L = price_diff × units (quote currency), then convert to JPY
        if the quote currency is not JPY.

        Args:
            trade_info: Stored trade information.
            exit_price: The exit price to use for the estimate.

        Returns:
            Estimated profit/loss in account currency (JPY).
        """
        try:
            instrument = trade_info.get("instrument", "")
            entry = trade_info.get("entry_price", 0)
            size = trade_info.get("size", 0)
            direction = trade_info.get("direction", "BUY")

            if exit_price == 0 or entry == 0 or size == 0:
                return 0.0

            if direction == "BUY":
                price_diff = exit_price - entry
            else:
                price_diff = entry - exit_price

            # P&L in quote currency = price_diff × units
            pnl = price_diff * size

            # Convert to account currency (JPY) if quote is not JPY
            parts = instrument.split("_")
            quote_ccy = parts[1] if len(parts) == 2 else ""
            if quote_ccy != "JPY" and hasattr(self.broker, "_get_conversion_rate"):
                pnl *= self.broker._get_conversion_rate(quote_ccy, "JPY")

            return pnl

        except Exception:
            return 0.0

    # --------------------------------------------------
    # Utility Methods
    # --------------------------------------------------

    def run_backtest(
        self,
        instrument: str,
        resolution: str = "HOUR_4",
        external_df: pd.DataFrame | None = None,
        confidence_threshold: int = 50,
    ) -> dict:
        """Run a backtest for a specific pair.

        Args:
            instrument: OANDA instrument.
            resolution: Data resolution.
            external_df: External OHLCV DataFrame (e.g., from CSV/yfinance).
                         If provided, skips API data fetch.

        Returns:
            Backtest results dictionary.
        """
        from src.models.backtest_engine import BacktestEngine

        # Use external data or fetch from API
        if external_df is not None and not external_df.empty:
            df = external_df.copy()
            logger.info(f"Running backtest for {instrument} with external data ({len(df)} candles)...")
        else:
            logger.info(f"Running backtest for {instrument} from API...")
            if not self.broker._connected:
                self.broker.connect()
            df = self.data_manager.fetch_prices(instrument, resolution, num_points=5000)

        if df.empty:
            logger.error("No data for backtest")
            return {}

        # Add indicators
        df = self.tech_analyzer.add_all_indicators(df)

        # Detect patterns on all bars
        # MIN_SIGNAL_GAP: same pattern+direction must be at least this many bars apart.
        # Prevents the same pattern from firing 30+ times as the sliding window passes over it.
        MIN_SIGNAL_GAP = 20  # 20 × 4H = 80 hours minimum between same signals

        signals = []
        last_signal_bar: dict[str, int] = {}  # key: "pattern_direction" -> last bar_index
        window_size = 100
        for i in range(window_size, len(df)):
            window = df.iloc[i - window_size:i + 1]
            patterns = self.pattern_detector.detect_all_patterns(window)
            for p in patterns:
                if not p.is_valid or p.confidence < confidence_threshold:
                    continue
                key = f"{p.pattern.value}_{p.direction.value}"
                last_bar = last_signal_bar.get(key, -MIN_SIGNAL_GAP - 1)
                if i - last_bar < MIN_SIGNAL_GAP:
                    continue  # same pattern fired too recently — skip duplicate
                signals.append({
                    "bar_index": i,
                    "direction": p.direction.value,
                    "entry_price": p.entry_price,
                    "stop_loss": p.stop_loss,
                    "take_profit": p.take_profit,
                    "pattern": p.pattern.value,
                })
                last_signal_bar[key] = i

        logger.info(f"Generated {len(signals)} signals for backtest (deduped, gap={MIN_SIGNAL_GAP} bars)")

        # Run backtest
        pair_config = self.config.pair_configs.get(instrument, {})
        pip_value = self.broker.get_pip_value(instrument) if hasattr(self.broker, 'get_pip_value') else pair_config.get("pip_value", 0.01)

        bt_config = dict(self.config.backtesting)
        bt_config["adaptive"] = self.config.risk_management.get("adaptive", {})
        engine = BacktestEngine(bt_config)

        # Run FIXED risk backtest
        result_fixed = engine.run(df, signals, pip_value, adaptive=False)

        # Run ADAPTIVE risk backtest
        result_adaptive = engine.run(df, signals, pip_value, adaptive=True)

        # --- Pretty print results ---
        period_start = df.index[0].strftime("%Y-%m-%d")
        period_end = df.index[-1].strftime("%Y-%m-%d")
        initial_balance = self.broker._paper_balance

        ra = result_adaptive
        rf = result_fixed

        print()
        print("=" * 62)
        print(f"  {instrument}  バックテスト結果")
        print(f"  期間: {period_start} ~ {period_end}  ({len(df):,}本 / {resolution})")
        print(f"  元金: ¥{initial_balance:,.0f}")
        print("=" * 62)
        print()
        print(f"  {'':30s} {'通常':>10s}   {'適応制御':>10s}")
        print(f"  {'-'*30} {'-'*10}   {'-'*10}")
        print(f"  {'トレード数':30s} {rf.total_trades:>10d}   {ra.total_trades:>10d}")
        print(f"  {'勝ちトレード':30s} {rf.winning_trades:>10d}   {ra.winning_trades:>10d}")
        print(f"  {'負けトレード':30s} {rf.losing_trades:>10d}   {ra.losing_trades:>10d}")
        print(f"  {'勝率':30s} {rf.win_rate:>9.1%}   {ra.win_rate:>9.1%}")
        print()
        print(f"  {'損益合計':30s} {'¥' + f'{rf.total_pnl:+,.0f}':>10s}   {'¥' + f'{ra.total_pnl:+,.0f}':>10s}")
        print(f"  {'リターン':30s} {rf.total_pnl / initial_balance:>+9.1%}   {ra.total_pnl / initial_balance:>+9.1%}")
        print(f"  {'平均利益':30s} {'¥' + f'{rf.avg_win:,.0f}':>10s}   {'¥' + f'{ra.avg_win:,.0f}':>10s}")
        print(f"  {'平均損失':30s} {'¥' + f'{rf.avg_loss:,.0f}':>10s}   {'¥' + f'{ra.avg_loss:,.0f}':>10s}")
        print()
        print(f"  {'プロフィットファクター (PF)':30s} {rf.profit_factor:>10.2f}   {ra.profit_factor:>10.2f}")
        print(f"  {'シャープレシオ':30s} {rf.sharpe_ratio:>10.2f}   {ra.sharpe_ratio:>10.2f}")
        print(f"  {'最大ドローダウン':30s} {rf.max_drawdown_pct:>9.1%}   {ra.max_drawdown_pct:>9.1%}")
        print(f"  {'最大連勝':30s} {rf.max_consecutive_wins:>10d}   {ra.max_consecutive_wins:>10d}")
        print(f"  {'最大連敗':30s} {rf.max_consecutive_losses:>10d}   {ra.max_consecutive_losses:>10d}")
        print()

        # Verdict
        fixed_ok = rf.meets_criteria()
        adaptive_ok = ra.meets_criteria()
        print(f"  AI仙人基準 (PF>=1.2, 勝率>=35%)")
        print(f"    通常:     {'PASS' if fixed_ok else 'FAIL'}")
        print(f"    適応制御: {'PASS' if adaptive_ok else 'FAIL'}")

        fixed_dd = rf.max_drawdown_pct
        adaptive_dd = ra.max_drawdown_pct
        if fixed_dd > 0:
            dd_improvement = (1 - adaptive_dd / fixed_dd) * 100
            print(f"    DD改善率: {dd_improvement:+.1f}%")

        print("=" * 62)

        # Return adaptive result as primary (with fixed for reference)
        fixed_s = rf.summary()
        adaptive_s = ra.summary()
        combined = adaptive_s.copy()
        combined["fixed_comparison"] = fixed_s
        return combined

    def get_status(self) -> dict[str, Any]:
        """Get current bot status.

        Returns:
            Status dictionary.
        """
        account = self.broker.get_account_info()
        return {
            "running": self._running,
            "mode": "LIVE" if self.config.is_live else "PAPER",
            "data_source": self.config.data_source,
            "account": account,
            "risk": self.risk_manager.get_risk_summary(),
            "open_trades": self._open_trades,
            "active_pairs": self.config.active_pairs,
            "timestamp": datetime.now().isoformat(),
        }
