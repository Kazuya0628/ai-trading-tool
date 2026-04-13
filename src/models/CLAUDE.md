# モデル層 - リスク管理・バックテスト

## リスク管理パラメータ（`risk_manager.py`）

| パラメータ | 値 | 設定キー |
| --- | --- | --- |
| 1トレードリスク | 口座の1% | `risk_per_trade_pct` |
| ストップロス | ATR × 2.0 | `stop_loss_atr_multiple` |
| テイクプロフィット | R:R 2:1 | `take_profit_rr_ratio` |
| 最小R:R | 0.5 | `min_risk_reward_ratio` |
| 最大ドローダウン | 10%で全停止 | `max_drawdown_pct` |
| デイリー損失上限 | 2% | `max_daily_loss_pct` |
| ウィークリー損失上限 | 5% | `max_weekly_loss_pct` |
| 最大同時ポジション | 3 | `max_open_positions` |
| トレーリングストップ | ATR × 1.5 | `trailing_stop_atr_multiple` |

## 適応的リスク制御（AI駆動）

| 要因 | 動作 | 設定キー |
| --- | --- | --- |
| 連敗スケーリング | 連敗ごとにリスク×0.75（最低25%まで） | `adaptive.loss_scale_factor` |
| ドローダウン防御 | DD 3%→×0.75、5%→×0.50、7%→×0.25 | `adaptive.drawdown_thresholds` |
| AI市場レジーム | Geminiがトレンド/レンジ/高ボラを判定し倍率設定 | AI自動判定 |
| 勝率モメンタム | 直近20トレードの勝率に応じて調整 | `adaptive.performance_window` |
| パターン無効化 | 勝率25%未満のパターンを自動スキップ | `adaptive.pattern_disable_win_rate` |
| 強制クールダウン | 5連敗で一時停止 | `adaptive.max_consecutive_losses` |

```
最終リスク = 基本リスク(1%) × 連敗係数 × DD係数 × レジーム係数 × 勝率係数
例: 1% × 0.75 × 0.50 × 0.8 × 0.85 = 0.255%（大幅縮小）
```

## バックテスト結果（Fixed vs Adaptive）

`--backtest` で Fixed（従来）と Adaptive（適応的）を自動比較表示。

| 指標 | USD/JPY Fixed→Adaptive | EUR/USD Fixed→Adaptive | GBP/JPY Fixed→Adaptive |
| --- | :---: | :---: | :---: |
| 勝率 | 52.5% → **67.7%** | 41.2% → **68.0%** | 52.8% → **73.8%** |
| PF | 2.13 → **5.70** | 1.11 → **5.34** | 1.81 → **4.76** |
| 最大DD | 43.4% → **8.3%** | 49.7% → **6.0%** | 60.2% → **6.8%** |
| シャープ | 3.47 → **6.19** | 0.57 → **6.98** | 2.61 → **6.34** |
| 最大連敗 | 32 → **11** | 42 → **14** | 35 → **20** |
