"""Bulk backtest for additional OANDA currency pairs."""
import sys
sys.path.insert(0, ".")

from src.trading_bot import TradingBot

bot = TradingBot()
bot.broker.connect()

pairs = [
    "AUD_USD", "NZD_USD", "USD_CHF", "USD_CAD",
    "AUD_JPY", "NZD_JPY", "CAD_JPY", "CHF_JPY",
    "EUR_GBP", "EUR_AUD", "EUR_CAD", "EUR_CHF", "EUR_NZD",
    "GBP_AUD", "GBP_CAD", "GBP_CHF", "GBP_NZD",
    "AUD_NZD", "AUD_CAD", "AUD_CHF",
    "NZD_CAD", "NZD_CHF", "CAD_CHF",
]

results = []
for pair in pairs:
    try:
        r = bot.run_backtest(pair, "HOUR_4")
        if r:
            pnl = float(str(r.get("total_pnl", "0")).replace(",", ""))
            pf = float(str(r.get("profit_factor", "0")))
            wr = r.get("win_rate", "0%")
            dd = r.get("max_drawdown_pct", "0%")
            sr = float(str(r.get("sharpe_ratio", "0")))
            trades = r.get("total_trades", 0)
            results.append({
                "pair": pair, "trades": trades, "wr": wr, "pf": pf,
                "pnl": pnl, "dd": dd, "sr": sr,
            })
    except Exception as e:
        print(f"ERROR {pair}: {e}")

bot.broker.disconnect()

# Sort by profit factor descending
results.sort(key=lambda x: x["pf"], reverse=True)

print()
print("=" * 75)
print("  追加ペア バックテスト結果（適応制御 / 4H足 / 約3.3年）")
print("=" * 75)
header = f"  {'ペア':<12} {'トレード':>6} {'勝率':>8} {'PF':>6} {'損益':>14} {'最大DD':>8} {'SR':>7}"
print(header)
print("  " + "-" * 71)
for r in results:
    pnl_str = "¥{:+,.0f}".format(r["pnl"])
    print(f"  {r['pair']:<12} {r['trades']:>6} {r['wr']:>8} {r['pf']:>6.2f} {pnl_str:>14} {r['dd']:>8} {r['sr']:>7.2f}")

# Highlight good ones
print()
print("  --- PF >= 1.2 かつ 勝率 >= 35% のペア ---")
good = [r for r in results if r["pf"] >= 1.2 and float(r["wr"].replace("%", "")) >= 35]
if good:
    for r in good:
        pnl_str = "¥{:+,.0f}".format(r["pnl"])
        print(f"  {r['pair']:<12} PF={r['pf']:.2f}  勝率={r['wr']}  損益={pnl_str}  DD={r['dd']}  SR={r['sr']:.2f}")
else:
    print("  なし")
print("=" * 75)
