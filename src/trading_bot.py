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
from typing import Any

import pandas as pd
from loguru import logger

from src.data.indicators import TechnicalAnalyzer
from src.data.market_data import MarketDataManager
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

        # Initialize risk manager
        self.risk_manager = RiskManager(self.config.risk_management)

        # Initialize notifier (Email / Discord / LINE)
        self.notifier = Notifier(
            discord_webhook=self.config.env.get("DISCORD_WEBHOOK_URL", ""),
            line_token=self.config.env.get("LINE_NOTIFY_TOKEN", ""),
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

    def _update_market_regime(self) -> None:
        """Run AI market regime analysis to update adaptive risk parameters."""
        if not self.ai_analyzer:
            return

        try:
            # Use first active pair's primary data for regime analysis
            instrument = self.config.active_pairs[0]
            resolution = self.config.timeframes.get("primary", "HOUR_4")
            df = self.data_manager.fetch_prices(instrument, resolution, num_points=200)

            if df.empty:
                return

            df = self.tech_analyzer.add_all_indicators(df)
            regime_data = self.ai_analyzer.analyze_market_regime(df, instrument)

            # Update the adaptive controller's market regime
            from src.models.risk_manager import MarketRegime
            self.risk_manager.adaptive.market_regime = MarketRegime(
                regime=regime_data.get("regime", "normal"),
                volatility_level=regime_data.get("volatility_level", "medium"),
                trend_strength=regime_data.get("trend_strength", "moderate"),
                risk_multiplier=regime_data.get("risk_multiplier", 0.8),
                confidence=regime_data.get("confidence", 50),
                reasoning=regime_data.get("reasoning", ""),
            )

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

        # Check if already have position in this pair
        if instrument in self._open_trades:
            logger.debug(f"Already have position in {pair_name}, skipping")
            return

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

        # 3. Take the best signal (with adaptive pattern filtering)
        best_signal = None
        for sig in signals:
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

        # 6. Execute trade (always paper mode)
        self._execute_trade(instrument, pair_name, best_signal, pos_size)

    def _execute_trade(
        self,
        instrument: str,
        pair_name: str,
        signal: TradeSignal,
        pos_size: Any,
    ) -> None:
        """Execute a paper trade.

        Args:
            instrument: OANDA instrument.
            pair_name: Currency pair name.
            signal: Trade signal.
            pos_size: Position size calculation.
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
            self._open_trades[instrument] = {
                "deal_id": deal_id,
                "instrument": instrument,
                "direction": signal.direction.value,
                "entry_price": result.get("level", signal.entry_price),
                "stop_loss": signal.stop_loss,
                "take_profit": signal.take_profit,
                "size": pos_size.lots,
                "pattern": signal.pattern.value,
                "opened_at": datetime.now().isoformat(),
            }

            log_trade(
                f"OPEN|{pair_name}|{signal.direction.value}|"
                f"deal={deal_id}|"
                f"entry={signal.entry_price:.5f}|"
                f"sl={signal.stop_loss:.5f}|"
                f"tp={signal.take_profit:.5f}|"
                f"size={pos_size.lots:.2f}|"
                f"pattern={signal.pattern.value}"
            )

            self.notifier.trade_opened(
                pair=pair_name,
                direction=signal.direction.value,
                size=pos_size.lots,
                price=signal.entry_price,
                sl=signal.stop_loss,
                tp=signal.take_profit,
                pattern=signal.pattern.value,
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

                # Notify
                self.notifier.trade_closed(
                    pair=pair_name,
                    direction=trade_info.get("direction", ""),
                    pnl=pnl,
                    pips=0,
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
                        self.broker.update_position(
                            deal_id=trade_info["deal_id"],
                            stop_level=new_sl,
                        )
                        trade_info["stop_loss"] = new_sl
                        logger.info(
                            f"Trailing stop updated: {instrument} "
                            f"new SL={new_sl:.5f}"
                        )

            except Exception as e:
                logger.error(f"Error monitoring {instrument}: {e}")

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
        signals = []
        window_size = 100
        for i in range(window_size, len(df)):
            window = df.iloc[i - window_size:i + 1]
            patterns = self.pattern_detector.detect_all_patterns(window)
            for p in patterns:
                if p.is_valid and p.confidence >= 50:
                    signals.append({
                        "bar_index": i,
                        "direction": p.direction.value,
                        "entry_price": p.entry_price,
                        "stop_loss": p.stop_loss,
                        "take_profit": p.take_profit,
                        "pattern": p.pattern.value,
                    })

        logger.info(f"Generated {len(signals)} signals for backtest")

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
