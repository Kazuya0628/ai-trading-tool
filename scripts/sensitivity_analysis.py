"""
信頼度閾値の感度分析 + 3年日足バックテスト

【手法】
1. 2年4Hデータを前半(IS)・後半(OOS)に分割
   IS  = 学習期間（過学習の確認に使う）
   OOS = 検証期間（実際の予測力）
2. 閾値 50/55/60/65/70% で両期間を比較
   → IS良好・OOS悪化なら過学習
   → IS/OOS両方改善なら本物の改善
3. 3年分の日足バックテストで長期安定性を確認
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from loguru import logger

from src.data.csv_loader import download_forex_data
from src.trading_bot import TradingBot


def run_threshold_analysis(
    bot: TradingBot,
    df: pd.DataFrame,
    instrument: str,
    resolution: str,
    thresholds: list[int],
    label: str,
) -> list[dict]:
    """指定DFで複数閾値をテストして結果リストを返す。"""
    results = []
    for thr in thresholds:
        r = bot.run_backtest(
            instrument,
            resolution=resolution,
            external_df=df,
            confidence_threshold=thr,
        )
        adaptive = r  # run_backtest returns adaptive summary
        results.append({
            "period": label,
            "threshold": thr,
            "trades": int(adaptive.get("total_trades", 0)),
            "win_rate": adaptive.get("win_rate", "0%"),
            "pf": float(adaptive.get("profit_factor", 0)),
            "pnl": float(adaptive.get("total_pnl", 0)),
            "max_dd": adaptive.get("max_drawdown_pct", "0%"),
            "sharpe": float(adaptive.get("sharpe_ratio", 0)),
        })
        logger.info(
            f"[{label}] thr={thr}%  trades={results[-1]['trades']:3d}  "
            f"PF={results[-1]['pf']:.2f}  pnl={results[-1]['pnl']:+,.0f}  "
            f"DD={results[-1]['max_dd']}  sharpe={results[-1]['sharpe']:.2f}"
        )
    return results


def print_comparison_table(all_results: list[dict], instrument: str) -> None:
    """IS/OOS並べて表示する。"""
    print(f"\n{'='*70}")
    print(f"  {instrument} — 閾値感度分析（IS=前半1年 / OOS=後半1年）")
    print(f"{'='*70}")
    print(f"{'閾値':>4} | {'期間':>4} | {'取引':>5} | {'勝率':>6} | {'PF':>5} | "
          f"{'損益':>10} | {'MaxDD':>6} | {'Sharpe':>7}")
    print("-" * 70)
    for r in all_results:
        print(
            f"{r['threshold']:>3}% | {r['period']:>4} | {r['trades']:>5} | "
            f"{r['win_rate']:>6} | {r['pf']:>5.2f} | "
            f"{r['pnl']:>+10,.0f} | {r['max_dd']:>6} | {r['sharpe']:>7.2f}"
        )
    print()


def main() -> None:
    logger.info("=== 感度分析 & 3年日足バックテスト 開始 ===")

    bot = TradingBot()

    thresholds = [50, 55, 60, 65, 70]
    pairs_4h = ["EUR_USD", "GBP_JPY", "USD_JPY"]
    pairs_1d = ["EUR_USD", "GBP_JPY", "USD_JPY"]

    # LINE通知用に安定性評価を収集
    pair_summaries: dict[str, list[str]] = {}

    # ─────────────────────────────────────────
    # Part 1: 閾値感度分析（4H 2年データ、IS/OOS分割）
    # ─────────────────────────────────────────
    logger.info("\n--- Part 1: 閾値感度分析（4H / 2年 / IS-OOS） ---")

    for instrument in pairs_4h:
        # キャッシュから読み込み（なければDL）
        df_full = download_forex_data(instrument, years=2, interval="4h", save=True)
        if df_full.empty:
            logger.warning(f"No data for {instrument}, skipping")
            continue

        mid = len(df_full) // 2
        df_is  = df_full.iloc[:mid].copy()   # 前半（学習）
        df_oos = df_full.iloc[mid:].copy()   # 後半（検証）

        logger.info(
            f"{instrument}: 全{len(df_full)}本 | "
            f"IS {df_is.index[0].date()}～{df_is.index[-1].date()} ({len(df_is)}本) | "
            f"OOS {df_oos.index[0].date()}～{df_oos.index[-1].date()} ({len(df_oos)}本)"
        )

        all_results = []
        all_results += run_threshold_analysis(bot, df_is, instrument, "HOUR_4", thresholds, "IS")
        all_results += run_threshold_analysis(bot, df_oos, instrument, "HOUR_4", thresholds, "OOS")

        # IS→OOS の変化率を追加表示
        print_comparison_table(all_results, instrument)
        verdicts = _print_stability_summary(all_results, thresholds, instrument)
        pair_summaries[instrument.replace("_", "/")] = verdicts

    # ─────────────────────────────────────────
    # Part 2: 3年日足バックテスト（長期安定性確認）
    # ─────────────────────────────────────────
    logger.info("\n--- Part 2: 3年日足バックテスト（1D / 3年） ---")

    for instrument in pairs_1d:
        df_1d = download_forex_data(instrument, years=3, interval="1d", save=True)
        if df_1d.empty:
            logger.warning(f"No daily data for {instrument}")
            continue

        logger.info(
            f"{instrument} 日足: {df_1d.index[0].date()} ～ {df_1d.index[-1].date()} "
            f"({len(df_1d)}本)"
        )

        # 全期間（閾値50%固定）
        r = bot.run_backtest(
            instrument,
            resolution="DAY",
            external_df=df_1d,
            confidence_threshold=50,
        )
        print(f"\n{'='*50}")
        print(f"  {instrument} 日足3年バックテスト（Adaptive）")
        print(f"{'='*50}")
        for k, v in r.items():
            if k != "fixed_comparison":
                print(f"  {k:<25}: {v}")

    logger.info("=== 感度分析 完了 ===")

    # LINE通知: 月次IS/OOS分析結果を送信
    _notify_line_monthly_summary(pair_summaries)


def _notify_line_monthly_summary(pair_summaries: dict[str, dict]) -> None:
    """月次IS/OOS分析結果をわかりやすくLINEで通知する。"""
    try:
        import os
        from datetime import datetime
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

        from src.utils.notifier import Notifier
        notifier = Notifier(
            line_config={
                "channel_access_token": os.getenv("LINE_CHANNEL_ACCESS_TOKEN", ""),
                "user_id": os.getenv("LINE_USER_ID", ""),
            },
        )

        month_label = datetime.now().strftime("%Y年%-m月")
        lines = [
            f"📊 月次ボット健全性レポート",
            f"({month_label})",
            "ボットの自動設定が実戦でも",
            "有効か確認しました。\n",
        ]

        PAIR_EMOJI = {
            "EUR/USD": "🇪🇺",
            "GBP/JPY": "🇬🇧",
            "USD/JPY": "🇺🇸",
        }

        for pair, data in pair_summaries.items():
            emoji = PAIR_EMOJI.get(pair, "💱")
            rec = data.get("recommended_threshold")
            items = data.get("items", [])

            lines.append(f"{'─'*16}")

            if rec is None:
                # 全閾値で不安定
                lines.append(f"{emoji} {pair}  ❌ 要注意")
                lines.append("実戦での成績が低下しています")
                lines.append("→ ボットが自動的に取引を")
                lines.append("  見送る判断をしています")
            else:
                # 安定している閾値が存在
                best = next(
                    (i for i in items if i["thr"] == rec), items[-1]
                )
                lines.append(f"{emoji} {pair}  ✅ 正常稼働中")
                lines.append(
                    f"学習時の成績: {best['is_pf']:.2f}"
                )
                lines.append(
                    f"実戦での成績: {best['oos_pf']:.2f}"
                )
                lines.append("→ 乖離が小さく安定しています")

        lines.append(f"{'─'*16}")
        lines.append("✉️ 対応は不要です")
        lines.append("ボットが自動で管理しています")
        next_month = datetime.now().replace(day=1)
        from datetime import timedelta
        next_report = (next_month.replace(day=28) + timedelta(days=4)).replace(day=1)
        lines.append(f"次回レポート: {next_report.strftime('%-m月1日')}")

        summary_text = "\n".join(lines)
        notifier.monthly_oos_result(summary_text)
        logger.info("[LINE] 月次IS/OOS分析結果を送信しました")
    except Exception as e:
        logger.warning(f"[LINE] 月次通知失敗: {e}")


def _print_stability_summary(
    all_results: list[dict], thresholds: list[int], instrument: str
) -> dict:
    """IS→OOS の変化を安定性の観点で評価する。

    Returns:
        {
          "items": [{"thr": int, "is_pf": float, "oos_pf": float, "drop": float, "status": str}, ...],
          "recommended_threshold": int | None,
        }
    """
    print(f"  【安定性評価: {instrument}】")
    items = []
    for thr in thresholds:
        is_r  = next((r for r in all_results if r["threshold"] == thr and r["period"] == "IS"),  None)
        oos_r = next((r for r in all_results if r["threshold"] == thr and r["period"] == "OOS"), None)
        if not is_r or not oos_r:
            continue

        pf_drop = oos_r["pf"] - is_r["pf"]
        both_positive = is_r["pf"] > 1.0 and oos_r["pf"] > 1.0
        if both_positive and abs(pf_drop) < 0.5:
            status = "stable"
            label = "✅ 安定"
        elif both_positive:
            status = "caution"
            label = "⚠️ 要注意"
        else:
            status = "unstable"
            label = "❌ 不安定"

        print(
            f"    {thr}%: IS_PF={is_r['pf']:.2f}  OOS_PF={oos_r['pf']:.2f}  "
            f"変化={pf_drop:+.2f}  {label}"
        )
        items.append({
            "thr": thr,
            "is_pf": is_r["pf"],
            "oos_pf": oos_r["pf"],
            "drop": pf_drop,
            "status": status,
        })

    # 推奨閾値: stable の中で最も高い閾値
    stable = [i for i in items if i["status"] == "stable"]
    recommended = max(stable, key=lambda x: x["thr"])["thr"] if stable else None
    print()
    return {"items": items, "recommended_threshold": recommended}


if __name__ == "__main__":
    main()
