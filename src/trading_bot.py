"""Automated Trading Bot - Main execution engine.

Combines all modules into a fully automated trading system:
1. Connect to data source (Twelve Data or OANDA)
2. Fetch multi-timeframe data
3. Calculate indicators
4. Detect patterns & AI analysis
5. Risk validation
6. Execute paper trades
7. Monitor positions
8. Loop

"1 Person Hedge Fund" - AI Sennin
"""

from __future__ import annotations

import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from src.data.indicators import TechnicalAnalyzer
from src.data.market_data import MarketDataManager
from src.models.position_store import PositionStore
from src.models.risk_manager import RiskManager
from src.strategies.ai_analyzer import AIChartAnalyzer
from src.strategies.pattern_detector import PatternDetector, SignalDirection
from src.strategies.strategy_engine import StrategyEngine, TradeSignal
from src.utils.config_loader import TradingConfig, create_broker
from src.utils.logger import log_trade, setup_logging
from src.utils.notifier import Notifier


class TradingBot:
    """Fully automated FX trading bot.

    Supports multiple data sources (Twelve Data, OANDA) via
    DATA_SOURCE environment variable. Implements the complete
    trading loop with risk management, pattern detection,
    AI analysis, and paper trade execution.
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

        # Initialize broker client (Twelve Data or OANDA)
        self.broker = create_broker(self.config.env)

        # Initialize data manager
        self.data_manager = MarketDataManager(self.broker)

        # Initialize technical analyzer
        self.tech_analyzer = TechnicalAnalyzer(self.config.indicators)

        # Initialize pattern detector
        self.pattern_detector = PatternDetector(self.config.pattern_recognition)

        # Initialize AI analyzer (optional)
        self.ai_analyzer = None
        if self.config.ai_analysis.get("enabled") and self.config.env.get("GEMINI_API_KEY"):
            self.ai_analyzer = AIChartAnalyzer(
                api_key=self.config.env["GEMINI_API_KEY"],
                config=self.config.ai_analysis,
            )

        # Initialize strategy engine
        self.strategy_engine = StrategyEngine(
            pattern_detector=self.pattern_detector,
            ai_analyzer=self.ai_analyzer,
            config={
                "indicators": self.config.indicators,
            },
        )

        # Initialize position store (SQLite)
        self.position_store = PositionStore()

        # Initialize Groq reviewer（相場レジーム判断＋週次戦略レビュー）
        from src.strategies.groq_reviewer import GroqReviewer
        from src.data.sentiment_fetcher import SentimentFetcher
        self.groq_reviewer = GroqReviewer()
        self.sentiment_fetcher = SentimentFetcher()
        self._last_sentiment: dict = {}          # センチメントキャッシュ（1時間）
        self._last_sentiment_ts: float = 0.0

        # Initialize risk manager
        self.risk_manager = RiskManager(self.config.risk_management)

        # Initialize notifier (Email / Discord / LINE)
        self.notifier = Notifier(
            discord_webhook=self.config.env.get("DISCORD_WEBHOOK_URL", ""),
            line_token=self.config.env.get("LINE_NOTIFY_TOKEN", ""),
            line_config={
                "channel_access_token": self.config.env.get("LINE_CHANNEL_ACCESS_TOKEN", ""),
                "user_id": self.config.env.get("LINE_USER_ID", ""),
                "imgbb_api_key": self.config.env.get("IMGBB_API_KEY", ""),
            },
            email_config={
                "smtp_host": self.config.env.get("SMTP_HOST", ""),
                "smtp_port": self.config.env.get("SMTP_PORT", "587"),
                "smtp_user": self.config.env.get("SMTP_USER", ""),
                "smtp_password": self.config.env.get("SMTP_PASSWORD", ""),
                "from": self.config.env.get("EMAIL_FROM", ""),
                "to": self.config.env.get("EMAIL_TO", ""),
            },
        )

        # State tracking
        self._running = False
        self._open_trades: dict[str, dict[str, Any]] = {}
        self._last_summary_date: str = ""
        self._last_weekly_review_date: str = ""   # 週次閾値見直し用
        self._dd_warned_levels: set[float] = set()  # ドローダウン警告済みレベル
        self._prev_regime: str = ""                  # 前回の相場レジーム
        self._restore_open_trades()

    def _restore_open_trades(self) -> None:
        """Restore open positions from SQLite DB on startup."""
        try:
            positions = self.position_store.get_open_positions()
            if not positions:
                return

            for pos in positions:
                instrument = pos["instrument"]
                self._open_trades[instrument] = pos
                if hasattr(self.broker, "_paper_positions"):
                    self.broker._paper_positions[instrument] = {
                        "deal_id": pos.get("deal_id", ""),
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

            logger.info(
                f"Restored {len(positions)} open positions from DB: "
                f"{list(self._open_trades.keys())}"
            )
        except Exception as e:
            logger.error(f"Failed to restore positions from DB: {e}")

    def _has_open_position_in_log(self, instrument: str) -> bool:
        """Check if a position is open for the given instrument."""
        return self.position_store.has_open_position(instrument)

    # --------------------------------------------------
    # Main Trading Loop
    # --------------------------------------------------

    def start(self) -> None:
        """Start the automated trading loop."""
        logger.info("Starting trading bot...")

        # Connect to OANDA
        if not self.broker.connect():
            logger.error("Failed to connect to OANDA. Aborting.")
            return

        self._running = True
        interval = self.config.execution.get("check_interval_seconds", 300)

        # Initial account info
        account = self.broker.get_account_info()
        if account:
            logger.info(
                f"Account: {account.get('account_name', 'N/A')} | "
                f"Balance: ¥{account.get('balance', 0):,.0f} | "
                f"Available: ¥{account.get('available', 0):,.0f}"
            )
            self.risk_manager.state.peak_balance = account.get("balance", 0)

        self.notifier.alert(
            f"Trading bot started.\n"
            f"Mode: {'LIVE' if self.config.is_live else 'PAPER'}\n"
            f"Pairs: {', '.join(self.config.active_pairs)}"
        )

        try:
            while self._running:
                self._trading_cycle()
                logger.info(f"Next check in {interval}s...")
                time.sleep(interval)
        except KeyboardInterrupt:
            logger.info("Bot stopped by user (Ctrl+C)")
        except Exception as e:
            logger.error(f"Fatal error: {e}")
            logger.error(traceback.format_exc())
            self.notifier.alert(f"FATAL ERROR: {e}")
        finally:
            self.stop()

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

            self.notifier.send("週次戦略レビュー", msg)

        except Exception as e:
            logger.error(f"Weekly threshold review failed: {e}")

    def stop(self) -> None:
        """Stop the trading bot and disconnect."""
        self._running = False
        self.broker.disconnect()
        logger.info("Trading bot stopped")

    def _trading_cycle(self) -> None:
        """Execute one complete trading cycle.

        For each active currency pair:
        1. Fetch multi-timeframe data
        2. Calculate indicators
        3. Detect patterns & generate signals
        4. Validate against risk rules
        5. Execute approved trades
        6. Monitor open positions
        """
        logger.info("-" * 40)
        logger.info(f"Trading cycle: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Ensure session is alive
        self.broker.ensure_session()

        # Get account state
        account = self.broker.get_account_info()
        balance = account.get("balance", 0)

        # Check if trading is allowed
        can_trade, reason = self.risk_manager.can_trade(balance)
        if not can_trade:
            logger.warning(f"Trading paused: {reason}")
            return

        # Adaptive: check for forced cooldown after consecutive losses
        cooldown, cooldown_reason = self.risk_manager.adaptive.should_cooldown()
        if cooldown:
            logger.warning(f"Adaptive cooldown: {cooldown_reason}")
            return

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

        # Analyze each active pair
        for instrument in self.config.active_pairs:
            try:
                self._analyze_and_trade(instrument, balance)
            except Exception as e:
                logger.error(f"Error analyzing {instrument}: {e}")
                logger.debug(traceback.format_exc())

        # Log risk summary with adaptive info
        risk_summary = self.risk_manager.get_risk_summary()
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

    def _update_market_regime(self) -> None:
        """Groq + 外部センチメントで相場レジームを判断し、リスク管理に反映する。

        Geminiの画像ベース判断を廃止し、以下を組み合わせてGroqが判断:
          - テクニカル指標（ATR / ADX / RSI / MA / BB幅）
          - yfinanceニュース見出し
          - ForexFactory 経済指標カレンダー
          - Reddit r/Forex 投稿
        """
        import time as _time
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

    def _analyze_and_trade(self, instrument: str, balance: float) -> None:
        """Analyze a single currency pair and execute trades if conditions met.

        Args:
            instrument: OANDA instrument (e.g., 'USD_JPY').
            balance: Current account balance.
        """
        pair_config = self.config.pair_configs.get(instrument, {})
        pair_name = pair_config.get("name", instrument)
        pip_value = pair_config.get("pip_value", 0.01)

        logger.info(f"Analyzing {pair_name} ({instrument})...")

        # Check if already have position in this pair (triple check)
        # 1. In-memory check
        if instrument in self._open_trades:
            logger.debug(f"Already have position in {pair_name} (memory), skipping")
            return

        # 2. Trade log check (survives restarts)
        if self._has_open_position_in_log(instrument):
            logger.warning(
                f"Position found in trade log for {pair_name} but not in memory. "
                f"Restoring and skipping."
            )
            self._restore_open_trades()
            return

        # 3. Broker check (ground truth)
        try:
            broker_positions = self.broker.get_open_positions()
            if not broker_positions.empty and instrument in broker_positions.index:
                logger.warning(
                    f"Position found in broker for {pair_name} but not in memory. "
                    f"Skipping."
                )
                return
        except Exception:
            pass  # If broker check fails, rely on log check

        # 1. Fetch multi-timeframe data
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

        # 2. Run strategy engine
        signals = self.strategy_engine.analyze(
            epic=instrument,
            pair_name=pair_name,
            mtf_data=mtf_data,
            timeframe_config=timeframe_config,
        )

        if not signals:
            logger.info(f"No signals for {pair_name}")
            return

        # 3. Take the best signal (with adaptive pattern filtering + confidence threshold)
        thr = self.config.confidence_threshold
        best_signal = None
        for sig in signals:
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
        )

        adaptive_mult = self.risk_manager.adaptive.calculate_risk_multiplier(
            self.risk_manager.state.current_drawdown_pct
        )
        logger.info(
            f"Trade setup: {pair_name} {best_signal.direction.value} | "
            f"Entry={best_signal.entry_price:.5f} | "
            f"SL={best_signal.stop_loss:.5f} | "
            f"TP={best_signal.take_profit:.5f} | "
            f"Size={pos_size.lots:.2f} lots | "
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
        if instrument in self._open_trades or self._has_open_position_in_log(instrument):
            logger.warning(f"BLOCKED duplicate trade for {pair_name} at final check")
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
            size=pos_size.lots,
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
                "size": pos_size.lots,
                "pattern": signal.pattern.value,
                "opened_at": datetime.now().isoformat(),
            }
            self._open_trades[instrument] = trade_info
            self.position_store.save_position(trade_info)

            log_trade(
                f"OPEN|{pair_name}|{signal.direction.value}|"
                f"deal={deal_id}|"
                f"entry={signal.entry_price:.5f}|"
                f"sl={signal.stop_loss:.5f}|"
                f"tp={signal.take_profit:.5f}|"
                f"size={pos_size.lots:.2f}|"
                f"pattern={signal.pattern.value}"
            )

            # Send notification with chart image + Gemini analysis
            rr = abs(signal.take_profit - signal.entry_price) / abs(signal.entry_price - signal.stop_loss) if abs(signal.entry_price - signal.stop_loss) > 0 else 0
            self.notifier.send_with_chart(
                title=f"{signal.direction.value} {pair_name}",
                message=f"Pattern: {signal.pattern.value}",
                fields={
                    "Entry Price": f"{signal.entry_price:.5f}",
                    "Size": f"{pos_size.lots:.2f} lots",
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

    def _monitor_positions(self, positions_df: pd.DataFrame) -> None:
        """Monitor open positions for trailing stops and exits.

        Args:
            positions_df: DataFrame of open positions.
        """
        if positions_df.empty:
            # Check if we had tracked positions that are now closed
            for instrument, trade_info in list(self._open_trades.items()):
                # Estimate P&L from the closed position
                pnl = self._estimate_closed_pnl(trade_info)
                self.risk_manager.record_trade_result(
                    pnl=pnl,
                    pattern=trade_info.get("pattern", ""),
                    instrument=instrument,
                )

                # Get exit price for logging
                try:
                    price_info = self.data_manager.get_latest_price(instrument)
                    exit_price = price_info.get("mid", 0)
                except Exception:
                    exit_price = 0

                pair_name = instrument.replace("_", "/")
                self.position_store.close_position(
                    deal_id=trade_info.get("deal_id", ""),
                    exit_price=exit_price,
                    pnl=pnl,
                )

                log_trade(
                    f"CLOSE|{pair_name}|"
                    f"deal={trade_info.get('deal_id', '')}|"
                    f"exit={exit_price:.5f}|"
                    f"pnl={pnl:.2f}|"
                    f"pattern={trade_info.get('pattern', '')}"
                )

                logger.info(
                    f"Position closed: {instrument} "
                    f"deal={trade_info.get('deal_id', 'N/A')} "
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
                    pip_unit = 0.01 if "JPY" in instrument else 0.0001
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

                del self._open_trades[instrument]
            return

        # Update trailing stops for open positions
        for instrument, trade_info in self._open_trades.items():
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
                        )
                        trade_info["stop_loss"] = new_sl
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

    def fix_open_positions_sltp(self) -> None:
        """保有中の全ポジションのSL/TPを修正済みロジックで即時再計算・更新する。

        `python main.py --fix-sltp` から呼び出す。
        レジーム変化を待たずに今すぐ適用したい場合に使用。
        """
        if not self._open_trades:
            logger.info("[fix-sltp] 保有ポジションなし")
            return

        # 現在のレジームを最新状態に更新してから適用
        self._update_market_regime()
        current_regime = self.risk_manager.adaptive.market_regime.regime
        logger.info(
            f"[fix-sltp] 現在レジーム: {current_regime} | "
            f"{len(self._open_trades)}件のポジションを修正します"
        )
        self._adjust_open_positions_for_regime(current_regime)

    def _adjust_open_positions_for_regime(self, new_regime: str) -> None:
        """レジーム変化時に保有ポジションのSL/TPをレジームに合わせて再計算・更新する。

        SLは含み益を守る方向のみ移動（不利方向には動かさない）。
        TPは新レジームの期待値に合わせて再設定する。

        Args:
            new_regime: Groqが判断した新しい相場レジーム名。
        """
        if not self._open_trades:
            return

        logger.info(
            f"[Regime調整] レジーム変化({new_regime})により"
            f"{len(self._open_trades)}件の保有ポジションを見直します"
        )

        for instrument, trade_info in self._open_trades.items():
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

                # pip_size を価格から推定（JPY系=0.01、それ以外=0.0001）
                pip_size = 0.01 if entry_price > 50 else 0.0001

                # レジーム調整済みSL/TPを計算（min/max pip bounds 適用済み）
                new_sl = self.risk_manager.calculate_stop_loss(
                    direction=direction,
                    entry_price=entry_price,
                    atr=atr,
                    pip_size=pip_size,
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
                )

                # SQLiteに反映
                self.position_store.update_position(
                    deal_id=deal_id,
                    stop_loss=sl_to_apply,
                    take_profit=new_tp,
                )

                pair_name = instrument.replace("_", "/")
                decimals = 3 if "JPY" in instrument else 5

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

    def _estimate_closed_pnl(self, trade_info: dict[str, Any]) -> float:
        """Estimate P&L for a closed position.

        Checks the broker for realized P&L, or estimates from current price.

        Args:
            trade_info: Stored trade information.

        Returns:
            Estimated profit/loss in account currency.
        """
        try:
            instrument = trade_info.get("instrument", "")
            price_info = self.data_manager.get_latest_price(instrument) if instrument else {}
            current_price = price_info.get("mid", 0)

            if current_price == 0:
                return 0.0

            entry = trade_info.get("entry_price", 0)
            size = trade_info.get("size", 0)
            direction = trade_info.get("direction", "BUY")

            if direction == "BUY":
                pnl = (current_price - entry) * size * 10000
            else:
                pnl = (entry - current_price) * size * 10000

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
            df = self.data_manager.fetch_prices(instrument, resolution, num_points=500)

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
        pip_value = pair_config.get("pip_value", 0.01)

        bt_config = dict(self.config.backtesting)
        bt_config["adaptive"] = self.config.risk_management.get("adaptive", {})
        engine = BacktestEngine(bt_config)

        # Run FIXED risk backtest
        result_fixed = engine.run(df, signals, pip_value, adaptive=False)

        # Run ADAPTIVE risk backtest
        result_adaptive = engine.run(df, signals, pip_value, adaptive=True)

        logger.info("=" * 60)
        logger.info("BACKTEST COMPARISON: Fixed vs Adaptive Risk")
        logger.info("-" * 60)

        fixed_s = result_fixed.summary()
        adaptive_s = result_adaptive.summary()

        logger.info(f"  {'Metric':<25} {'Fixed':>12} {'Adaptive':>12}")
        logger.info(f"  {'-'*25} {'-'*12} {'-'*12}")
        for key in fixed_s:
            logger.info(f"  {key:<25} {str(fixed_s[key]):>12} {str(adaptive_s[key]):>12}")

        logger.info("-" * 60)
        logger.info(
            f"  Fixed meets criteria:    {'YES' if result_fixed.meets_criteria() else 'NO'}"
        )
        logger.info(
            f"  Adaptive meets criteria: {'YES' if result_adaptive.meets_criteria() else 'NO'}"
        )

        # Calculate drawdown improvement
        fixed_dd = result_fixed.max_drawdown_pct
        adaptive_dd = result_adaptive.max_drawdown_pct
        if fixed_dd > 0:
            dd_improvement = (1 - adaptive_dd / fixed_dd) * 100
            logger.info(f"  Drawdown improvement:    {dd_improvement:+.1f}%")

        logger.info("=" * 60)

        # Return adaptive result as primary (with fixed for reference)
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
