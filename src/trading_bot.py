"""Automated Trading Bot - Main execution engine.

Combines all modules into a fully automated trading system:
1. Connect to IG Securities
2. Fetch multi-timeframe data
3. Calculate indicators
4. Detect patterns & AI analysis
5. Risk validation
6. Execute trades
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

from src.brokers.ig_client import IGClient
from src.data.indicators import TechnicalAnalyzer
from src.data.market_data import MarketDataManager
from src.models.risk_manager import RiskManager
from src.strategies.ai_analyzer import AIChartAnalyzer
from src.strategies.pattern_detector import PatternDetector, SignalDirection
from src.strategies.strategy_engine import StrategyEngine, TradeSignal
from src.utils.config_loader import TradingConfig
from src.utils.logger import log_trade, setup_logging
from src.utils.notifier import Notifier


class TradingBot:
    """Fully automated FX trading bot for IG Securities.

    Implements the complete trading loop with risk management,
    pattern detection, AI analysis, and automated execution.
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
        logger.info(f"Account: {'DEMO' if self.config.is_demo else 'LIVE'}")
        logger.info("=" * 60)

        # Initialize IG Securities client
        self.ig_client = IGClient(
            api_key=self.config.env["IG_API_KEY"],
            username=self.config.env["IG_USERNAME"],
            password=self.config.env["IG_PASSWORD"],
            acc_type=self.config.env["IG_ACC_TYPE"],
            acc_number=self.config.env["IG_ACC_NUMBER"],
        )

        # Initialize data manager
        self.data_manager = MarketDataManager(self.ig_client)

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

        # Initialize notifier
        self.notifier = Notifier(
            discord_webhook=self.config.env.get("DISCORD_WEBHOOK_URL", ""),
            line_token=self.config.env.get("LINE_NOTIFY_TOKEN", ""),
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

        # Connect to IG Securities
        if not self.ig_client.connect():
            logger.error("Failed to connect to IG Securities. Aborting.")
            return

        self._running = True
        interval = self.config.execution.get("check_interval_seconds", 300)

        # Initial account info
        account = self.ig_client.get_account_info()
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
        self.ig_client.disconnect()
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
        self.ig_client.ensure_session()

        # Get account state
        account = self.ig_client.get_account_info()
        balance = account.get("balance", 0)

        # Check if trading is allowed
        can_trade, reason = self.risk_manager.can_trade(balance)
        if not can_trade:
            logger.warning(f"Trading paused: {reason}")
            return

        # Update open position count
        open_positions = self.ig_client.get_open_positions()
        if not open_positions.empty:
            self.risk_manager.update_open_positions(len(open_positions))
        else:
            self.risk_manager.update_open_positions(0)

        # Monitor existing positions (trailing stop, etc.)
        self._monitor_positions(open_positions)

        # Analyze each active pair
        for epic in self.config.active_pairs:
            try:
                self._analyze_and_trade(epic, balance)
            except Exception as e:
                logger.error(f"Error analyzing {epic}: {e}")
                logger.debug(traceback.format_exc())

        # Log risk summary
        risk_summary = self.risk_manager.get_risk_summary()
        logger.info(
            f"Risk: daily_pnl=¥{risk_summary['daily_pnl']:+,.0f} | "
            f"drawdown={risk_summary['drawdown_pct']:.1f}% | "
            f"trades={risk_summary['daily_trades']} | "
            f"positions={risk_summary['open_positions']}"
        )

    def _analyze_and_trade(self, epic: str, balance: float) -> None:
        """Analyze a single currency pair and execute trades if conditions met.

        Args:
            epic: IG market epic identifier.
            balance: Current account balance.
        """
        pair_config = self.config.pair_configs.get(epic, {})
        pair_name = pair_config.get("name", epic)
        pip_value = pair_config.get("pip_value", 0.01)

        logger.info(f"Analyzing {pair_name} ({epic})...")

        # Check if already have position in this pair
        if epic in self._open_trades:
            logger.debug(f"Already have position in {pair_name}, skipping")
            return

        # 1. Fetch multi-timeframe data
        timeframe_config = self.config.timeframes
        mtf_data = {}
        for tf_name in ["primary", "secondary", "trend"]:
            resolution = timeframe_config.get(tf_name)
            if resolution:
                df = self.data_manager.fetch_prices(epic, resolution, num_points=500)
                if not df.empty:
                    # Add indicators
                    df = self.tech_analyzer.add_all_indicators(df)
                    mtf_data[tf_name] = df

        if "primary" not in mtf_data:
            logger.warning(f"No primary data for {pair_name}")
            return

        # 2. Run strategy engine
        signals = self.strategy_engine.analyze(
            epic=epic,
            pair_name=pair_name,
            mtf_data=mtf_data,
            timeframe_config=timeframe_config,
        )

        if not signals:
            logger.info(f"No signals for {pair_name}")
            return

        # 3. Take the best signal
        best_signal = signals[0]

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

        logger.info(
            f"Trade setup: {pair_name} {best_signal.direction.value} | "
            f"Entry={best_signal.entry_price:.5f} | "
            f"SL={best_signal.stop_loss:.5f} | "
            f"TP={best_signal.take_profit:.5f} | "
            f"Size={pos_size.lots:.2f} lots | "
            f"Risk=¥{pos_size.risk_amount:,.0f} | "
            f"R:R={best_signal.risk_reward_ratio:.1f} | "
            f"Confidence={best_signal.confidence:.0f}%"
        )

        # 6. Execute trade
        if self.config.is_live:
            self._execute_trade(epic, pair_name, best_signal, pos_size)
        else:
            logger.info(f"[PAPER] Would execute: {best_signal.to_dict()}")
            log_trade(
                f"PAPER|{pair_name}|{best_signal.direction.value}|"
                f"entry={best_signal.entry_price:.5f}|"
                f"sl={best_signal.stop_loss:.5f}|"
                f"tp={best_signal.take_profit:.5f}|"
                f"size={pos_size.lots:.2f}|"
                f"pattern={best_signal.pattern.value}|"
                f"confidence={best_signal.confidence:.0f}"
            )

    def _execute_trade(
        self,
        epic: str,
        pair_name: str,
        signal: TradeSignal,
        pos_size: Any,
    ) -> None:
        """Execute a trade via IG Securities API.

        Args:
            epic: Market epic.
            pair_name: Currency pair name.
            signal: Trade signal.
            pos_size: Position size calculation.
        """
        result = self.ig_client.open_position(
            epic=epic,
            direction=signal.direction.value,
            size=pos_size.lots,
            order_type="MARKET",
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            trailing_stop=self.risk_manager.trailing_stop,
        )

        if result.get("success"):
            deal_id = result["deal_id"]
            self._open_trades[epic] = {
                "deal_id": deal_id,
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
            positions_df: DataFrame of open positions from IG.
        """
        if positions_df.empty:
            # Check if we had tracked positions that are now closed
            for epic, trade_info in list(self._open_trades.items()):
                logger.info(
                    f"Position closed (externally): {epic} "
                    f"deal={trade_info.get('deal_id', 'N/A')}"
                )
                del self._open_trades[epic]
            return

        # Update trailing stops for open positions
        for epic, trade_info in self._open_trades.items():
            if not self.risk_manager.trailing_stop:
                continue

            try:
                # Get current price
                price_info = self.data_manager.get_latest_price(epic)
                current_price = price_info.get("mid", 0)
                if current_price == 0:
                    continue

                # Get current ATR
                df = self.data_manager.fetch_prices(
                    epic,
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
                        # Update stop on IG
                        if self.config.is_live:
                            self.ig_client.update_position(
                                deal_id=trade_info["deal_id"],
                                stop_level=new_sl,
                            )
                        trade_info["stop_loss"] = new_sl
                        logger.info(
                            f"Trailing stop updated: {epic} "
                            f"new SL={new_sl:.5f}"
                        )

            except Exception as e:
                logger.error(f"Error monitoring {epic}: {e}")

    # --------------------------------------------------
    # Utility Methods
    # --------------------------------------------------

    def run_backtest(self, epic: str, resolution: str = "HOUR_4") -> dict:
        """Run a backtest for a specific pair.

        Args:
            epic: Market epic.
            resolution: Data resolution.

        Returns:
            Backtest results dictionary.
        """
        from src.models.backtest_engine import BacktestEngine

        logger.info(f"Running backtest for {epic}...")

        # Connect if needed
        if not self.ig_client._service:
            self.ig_client.connect()

        # Fetch historical data
        df = self.data_manager.fetch_prices(epic, resolution, num_points=500)
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
        pair_config = self.config.pair_configs.get(epic, {})
        pip_value = pair_config.get("pip_value", 0.01)

        engine = BacktestEngine(self.config.backtesting)
        result = engine.run(df, signals, pip_value)

        logger.info("=" * 40)
        logger.info("BACKTEST RESULTS")
        for k, v in result.summary().items():
            logger.info(f"  {k}: {v}")
        logger.info(
            f"  Meets criteria: {'YES ✓' if result.meets_criteria() else 'NO ✗'}"
        )
        logger.info("=" * 40)

        return result.summary()

    def get_status(self) -> dict[str, Any]:
        """Get current bot status.

        Returns:
            Status dictionary.
        """
        account = self.ig_client.get_account_info()
        return {
            "running": self._running,
            "mode": "LIVE" if self.config.is_live else "PAPER",
            "account_type": self.config.env.get("IG_ACC_TYPE", "DEMO"),
            "account": account,
            "risk": self.risk_manager.get_risk_summary(),
            "open_trades": self._open_trades,
            "active_pairs": self.config.active_pairs,
            "timestamp": datetime.now().isoformat(),
        }
