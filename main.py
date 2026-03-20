#!/usr/bin/env python3
"""AI FX Trading Tool - Main Entry Point.

Usage:
    python main.py                  # Start trading bot (paper mode)
    python main.py --live           # Start in live mode
    python main.py --backtest       # Run backtest
    python main.py --status         # Check account status
    python main.py --analyze EPIC   # Analyze a specific pair
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.trading_bot import TradingBot
from loguru import logger


def main() -> None:
    """Main entry point for the AI FX Trading Tool."""
    parser = argparse.ArgumentParser(
        description="AI FX Trading Tool - IG Securities Automated Trading",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                     Start in paper trading mode
  python main.py --live              Start in live trading mode
  python main.py --backtest          Run backtest on all active pairs
  python main.py --backtest --epic CS.D.USDJPY.TODAY.IP
                                     Run backtest on specific pair
  python main.py --status            Show account status
  python main.py --analyze CS.D.USDJPY.TODAY.IP
                                     Analyze a specific pair
        """,
    )

    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to trading_config.yaml",
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Enable live trading mode (USE WITH CAUTION)",
    )
    parser.add_argument(
        "--backtest", action="store_true",
        help="Run backtest instead of live trading",
    )
    parser.add_argument(
        "--epic", type=str, default=None,
        help="Specific epic for backtest/analysis",
    )
    parser.add_argument(
        "--resolution", type=str, default="HOUR_4",
        help="Data resolution for backtest (default: HOUR_4)",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Show account status and exit",
    )
    parser.add_argument(
        "--analyze", type=str, default=None,
        help="Analyze a specific epic and show signals",
    )

    args = parser.parse_args()

    # Initialize bot
    bot = TradingBot(config_path=args.config)

    if args.live:
        logger.warning("=" * 60)
        logger.warning("  ⚠️  LIVE TRADING MODE ENABLED  ⚠️")
        logger.warning("  Real money will be used for trades!")
        logger.warning("=" * 60)

        confirm = input("Type 'CONFIRM' to proceed with live trading: ")
        if confirm != "CONFIRM":
            logger.info("Live trading cancelled.")
            return

    # Execute mode
    if args.status:
        _show_status(bot)
    elif args.backtest:
        _run_backtest(bot, args.epic, args.resolution)
    elif args.analyze:
        _analyze_pair(bot, args.analyze)
    else:
        # Start trading bot
        bot.start()


def _show_status(bot: TradingBot) -> None:
    """Show account status."""
    if not bot.ig_client.connect():
        logger.error("Failed to connect")
        return

    status = bot.get_status()
    print("\n" + "=" * 50)
    print("  AI FX Trading Tool - Status")
    print("=" * 50)

    account = status.get("account", {})
    print(f"\n  Account: {account.get('account_name', 'N/A')}")
    print(f"  Balance: ¥{account.get('balance', 0):,.0f}")
    print(f"  P&L:     ¥{account.get('profit_loss', 0):+,.0f}")
    print(f"  Available: ¥{account.get('available', 0):,.0f}")
    print(f"  Type:    {status.get('account_type', 'N/A')}")

    risk = status.get("risk", {})
    print(f"\n  Daily P&L:    ¥{risk.get('daily_pnl', 0):+,.0f}")
    print(f"  Drawdown:     {risk.get('drawdown_pct', 0):.1f}%")
    print(f"  Open Pos:     {risk.get('open_positions', 0)}")
    print(f"  Trades Today: {risk.get('daily_trades', 0)}")

    print(f"\n  Active Pairs: {', '.join(status.get('active_pairs', []))}")
    print("=" * 50 + "\n")

    bot.ig_client.disconnect()


def _run_backtest(bot: TradingBot, epic: str | None, resolution: str) -> None:
    """Run backtest."""
    if epic:
        epics = [epic]
    else:
        epics = bot.config.active_pairs

    for e in epics:
        print(f"\n{'='*50}")
        print(f"  Backtest: {e} ({resolution})")
        print(f"{'='*50}")
        result = bot.run_backtest(e, resolution)
        if result:
            print(json.dumps(result, indent=2))

    bot.ig_client.disconnect()


def _analyze_pair(bot: TradingBot, epic: str) -> None:
    """Analyze a specific currency pair."""
    if not bot.ig_client.connect():
        logger.error("Failed to connect")
        return

    pair_config = bot.config.pair_configs.get(epic, {})
    pair_name = pair_config.get("name", epic)

    print(f"\n{'='*50}")
    print(f"  Analysis: {pair_name} ({epic})")
    print(f"{'='*50}")

    # Fetch data
    timeframe_config = bot.config.timeframes
    mtf_data = {}
    for tf_name in ["primary", "secondary", "trend"]:
        resolution = timeframe_config.get(tf_name)
        if resolution:
            df = bot.data_manager.fetch_prices(epic, resolution, num_points=500)
            if not df.empty:
                df = bot.tech_analyzer.add_all_indicators(df)
                mtf_data[tf_name] = df
                print(f"\n  {tf_name} ({resolution}): {len(df)} bars loaded")

    if "primary" not in mtf_data:
        print("  No data available")
        bot.ig_client.disconnect()
        return

    # Run analysis
    signals = bot.strategy_engine.analyze(
        epic=epic,
        pair_name=pair_name,
        mtf_data=mtf_data,
        timeframe_config=timeframe_config,
    )

    if signals:
        print(f"\n  Found {len(signals)} signal(s):")
        for i, s in enumerate(signals):
            print(f"\n  --- Signal #{i+1} ---")
            for k, v in s.to_dict().items():
                print(f"    {k}: {v}")
    else:
        print("\n  No signals detected")

    # Market info
    info = bot.ig_client.get_market_info(epic)
    if info:
        print(f"\n  Market Status:")
        print(f"    Bid:    {info.get('bid', 0):.5f}")
        print(f"    Offer:  {info.get('offer', 0):.5f}")
        print(f"    Spread: {info.get('spread', 0):.5f}")
        print(f"    Status: {info.get('market_status', 'N/A')}")

    print(f"\n{'='*50}\n")
    bot.ig_client.disconnect()


if __name__ == "__main__":
    main()
