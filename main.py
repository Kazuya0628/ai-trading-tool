#!/usr/bin/env python3
"""AI FX Trading Tool - Main Entry Point.

Data Source: Twelve Data / OANDA v20 API
Trading Mode: Paper trading (analysis & simulation only)

Usage:
    python main.py                  # Start trading bot (paper mode)
    python main.py --backtest       # Run backtest
    python main.py --status         # Check account status
    python main.py --analyze USD_JPY  # Analyze a specific pair
"""

from __future__ import annotations

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
        description="AI FX Trading Tool - OANDA Market Data + Paper Trading",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                     Start in paper trading mode
  python main.py --backtest          Run backtest on all active pairs
  python main.py --backtest --instrument USD_JPY
                                     Run backtest on specific pair
  python main.py --status            Show account status
  python main.py --analyze USD_JPY   Analyze a specific pair
        """,
    )

    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to trading_config.yaml",
    )
    parser.add_argument(
        "--backtest", action="store_true",
        help="Run backtest instead of live trading",
    )
    parser.add_argument(
        "--instrument", type=str, default=None,
        help="Specific instrument for backtest/analysis (e.g., USD_JPY)",
    )
    # Keep --epic as alias for backward compatibility
    parser.add_argument(
        "--epic", type=str, default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--resolution", type=str, default="HOUR_4",
        help="Data resolution for backtest (default: HOUR_4)",
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Path to CSV file for backtest (overrides API data)",
    )
    parser.add_argument(
        "--download", action="store_true",
        help="Download multi-year historical data via yfinance for backtest",
    )
    parser.add_argument(
        "--years", type=int, default=2,
        help="Years of data to download (default: 2, max 2 for 4h data)",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Show account status and exit",
    )
    parser.add_argument(
        "--analyze", type=str, default=None,
        help="Analyze a specific instrument and show signals (e.g., USD_JPY)",
    )
    parser.add_argument(
        "--fix-sltp", action="store_true",
        help="Recalculate and update SL/TP for all open positions immediately",
    )
    parser.add_argument(
        "--notify-analysis", action="store_true",
        help="Run analysis for all active pairs and send Gemini results to LINE immediately",
    )
    parser.add_argument(
        "--trade-once", action="store_true",
        help="Run one trading cycle immediately (analyze + execute trades if signals found)",
    )

    args = parser.parse_args()

    # Handle --epic alias
    instrument = args.instrument or args.epic

    # Initialize bot
    bot = TradingBot(config_path=args.config)

    # Execute mode
    if args.status:
        _show_status(bot)
    elif args.notify_analysis:
        bot.run_analysis_notify()
    elif args.backtest:
        _run_backtest(bot, instrument, args.resolution, args.csv, args.download, args.years)
    elif args.analyze:
        _analyze_pair(bot, args.analyze)
    elif args.fix_sltp:
        _fix_sltp(bot)
    elif args.trade_once:
        bot.run_trade_once()
    else:
        # Start trading bot (paper mode)
        bot.start()


def _show_status(bot: TradingBot) -> None:
    """Show account status."""
    if not bot.broker.connect():
        logger.error("Failed to connect to OANDA")
        return

    status = bot.get_status()
    print("\n" + "=" * 50)
    print("  AI FX Trading Tool - Status")
    print("=" * 50)

    account = status.get("account", {})
    print(f"\n  Data Source: {status.get('data_source', 'N/A')}")
    print(f"  Account: {account.get('account_name', 'N/A')}")
    print(f"  Balance: ¥{account.get('balance', 0):,.0f}")
    print(f"  P&L:     ¥{account.get('profit_loss', 0):+,.0f}")
    print(f"  Available: ¥{account.get('available', 0):,.0f}")

    risk = status.get("risk", {})
    print(f"\n  Daily P&L:    ¥{risk.get('daily_pnl', 0):+,.0f}")
    print(f"  Drawdown:     {risk.get('drawdown_pct', 0):.1f}%")
    print(f"  Open Pos:     {risk.get('open_positions', 0)}")
    print(f"  Trades Today: {risk.get('daily_trades', 0)}")

    print(f"\n  Active Pairs: {', '.join(status.get('active_pairs', []))}")
    print("=" * 50 + "\n")

    bot.broker.disconnect()


def _run_backtest(
    bot: TradingBot,
    instrument: str | None,
    resolution: str,
    csv_path: str | None = None,
    download: bool = False,
    years: int = 2,
) -> None:
    """Run backtest with optional CSV/download data source."""
    from src.data.csv_loader import download_forex_data, load_csv, load_or_download

    if instrument:
        instruments = [instrument]
    else:
        instruments = bot.config.active_pairs

    results: list[tuple[str, dict]] = []

    for inst in instruments:
        # Determine data source
        external_df = None

        if csv_path:
            # User-provided CSV file
            print(f"\n  Loading CSV: {csv_path}")
            external_df = load_csv(csv_path)
        elif download:
            # Download from yfinance
            external_df = load_or_download(inst, interval=resolution.lower().replace("hour_", "") + "h" if "HOUR" in resolution else "1d", years=years)

        if external_df is not None and not external_df.empty:
            result = bot.run_backtest(inst, resolution, external_df=external_df)
        else:
            result = bot.run_backtest(inst, resolution)

        if result:
            results.append((inst, result))

    # Print summary table if multiple pairs
    if len(results) > 1:
        print()
        print("=" * 80)
        print("  全ペア サマリー（適応制御）")
        print("=" * 80)
        print(f"  {'ペア':<10s} {'トレード':>6s} {'勝率':>7s} {'PF':>6s} {'損益':>14s} {'リターン':>8s} {'最大DD':>7s} {'SR':>6s}")
        print(f"  {'-'*10} {'-'*6} {'-'*7} {'-'*6} {'-'*14} {'-'*8} {'-'*7} {'-'*6}")
        total_pnl = 0.0
        for inst, r in results:
            pnl = float(r.get("total_pnl", "0").replace(",", ""))
            wr = r.get("win_rate", "0%")
            pf = r.get("profit_factor", "0")
            dd = r.get("max_drawdown_pct", "0%")
            sr = r.get("sharpe_ratio", "0")
            trades = r.get("total_trades", 0)
            balance = bot.broker._paper_balance
            ret = pnl / balance if balance > 0 else 0
            total_pnl += pnl
            print(f"  {inst:<10s} {trades:>6d} {wr:>7s} {pf:>6s} {'¥' + f'{pnl:+,.0f}':>14s} {ret:>+7.1%} {dd:>7s} {sr:>6s}")
        balance = bot.broker._paper_balance
        total_ret = total_pnl / balance if balance > 0 else 0
        print(f"  {'-'*10} {'-'*6} {'-'*7} {'-'*6} {'-'*14} {'-'*8} {'-'*7} {'-'*6}")
        print(f"  {'合計':<10s} {'':>6s} {'':>7s} {'':>6s} {'¥' + f'{total_pnl:+,.0f}':>14s} {total_ret:>+7.1%}")
        print("=" * 80)

    bot.broker.disconnect()


def _fix_sltp(bot: TradingBot) -> None:
    """Recalculate and apply corrected SL/TP to all open positions."""
    if not bot.broker.connect():
        logger.error("Failed to connect to broker")
        return

    print("\n" + "=" * 50)
    print("  SL/TP 修正モード")
    print("=" * 50)

    open_trades = bot._open_trades
    if not open_trades:
        print("\n  保有ポジションなし\n")
        bot.broker.disconnect()
        return

    print(f"\n  対象ポジション: {len(open_trades)}件")
    for inst, info in open_trades.items():
        decimals = 3 if "JPY" in inst else 5
        print(f"  {inst}: {info.get('direction')} "
              f"SL={info.get('stop_loss', 0):.{decimals}f} "
              f"TP={info.get('take_profit', 0):.{decimals}f}")

    print("\n  修正中...")
    bot.fix_open_positions_sltp()

    print("\n  修正後:")
    for inst, info in open_trades.items():
        decimals = 3 if "JPY" in inst else 5
        print(f"  {inst}: {info.get('direction')} "
              f"SL={info.get('stop_loss', 0):.{decimals}f} "
              f"TP={info.get('take_profit', 0):.{decimals}f}")

    print("\n" + "=" * 50 + "\n")
    bot.broker.disconnect()


def _analyze_pair(bot: TradingBot, instrument: str) -> None:
    """Analyze a specific currency pair."""
    if not bot.broker.connect():
        logger.error("Failed to connect to OANDA")
        return

    pair_config = bot.config.pair_configs.get(instrument, {})
    pair_name = pair_config.get("name", instrument)

    print(f"\n{'='*50}")
    print(f"  Analysis: {pair_name} ({instrument})")
    print(f"{'='*50}")

    # Fetch data
    timeframe_config = bot.config.timeframes
    mtf_data = {}
    for tf_name in ["primary", "secondary", "trend"]:
        resolution = timeframe_config.get(tf_name)
        if resolution:
            df = bot.data_manager.fetch_prices(instrument, resolution, num_points=500)
            if not df.empty:
                df = bot.tech_analyzer.add_all_indicators(df)
                mtf_data[tf_name] = df
                print(f"\n  {tf_name} ({resolution}): {len(df)} bars loaded")

    if "primary" not in mtf_data:
        print("  No data available")
        bot.broker.disconnect()
        return

    # Run analysis
    signals = bot.strategy_engine.analyze(
        epic=instrument,
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
    info = bot.broker.get_market_info(instrument)
    if info:
        print(f"\n  Market Status:")
        print(f"    Bid:    {info.get('bid', 0):.5f}")
        print(f"    Offer:  {info.get('offer', 0):.5f}")
        print(f"    Spread: {info.get('spread', 0):.5f}")
        print(f"    Status: {info.get('market_status', 'N/A')}")

    print(f"\n{'='*50}\n")
    bot.broker.disconnect()


if __name__ == "__main__":
    main()
