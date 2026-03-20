# AI FX Trading Tool - AI仙人メソドロジー自動売買システム

## プロジェクト概要
AI仙人のメソドロジーに基づくAI搭載FX分析・ペーパートレードツール。
デュアルブローカーアーキテクチャ（Twelve Data / OANDA）でリアルタイム為替データを取得し、
チャートパターン検出 + Gemini AIビジュアル分析によるシグナル生成・ペーパートレードを実行します。

## データソース

| ソース | 状態 | 用途 |
|--------|------|------|
| **Twelve Data** | 現在アクティブ | 無料枠（800リクエスト/日、8リクエスト/分） |
| **OANDA v20** | 将来切替用 | ライブ口座取得後に `.env` の `DATA_SOURCE=oanda` で切替可能 |

切替は `.env` の `DATA_SOURCE` を変更するだけで完了（コード変更不要）。

## アーキテクチャ

```
AiTradeTool/
├── main.py                      # エントリーポイント（CLI）
├── config/
│   └── trading_config.yaml      # トレード設定ファイル
├── src/
│   ├── trading_bot.py           # メイン自動売買エンジン
│   ├── brokers/
│   │   ├── oanda_client.py      # OANDA v20 API クライアント
│   │   └── twelvedata_client.py # Twelve Data API クライアント（現在使用中）
│   ├── data/
│   │   ├── market_data.py       # マーケットデータ取得・管理
│   │   ├── indicators.py        # テクニカル指標計算
│   │   └── csv_loader.py        # CSV/yfinanceデータローダー（長期BT用）
│   ├── models/
│   │   ├── risk_manager.py      # リスク管理・ポジションサイジング
│   │   └── backtest_engine.py   # バックテストエンジン
│   ├── strategies/
│   │   ├── pattern_detector.py  # チャートパターン検出（6パターン）
│   │   ├── ai_analyzer.py       # Gemini AI チャート分析
│   │   └── strategy_engine.py   # マルチ戦略エンジン
│   └── utils/
│       ├── config_loader.py     # 設定読み込み・ブローカーファクトリ
│       ├── logger.py            # ログ設定
│       └── notifier.py          # Email/Discord/LINE通知
├── deploy/
│   ├── install.sh               # ローカルインストールスクリプト
│   └── ai-fx-bot.service        # systemdサービス定義
├── tests/                       # テストスイート（46テスト）
├── data/
│   └── historical/              # yfinanceダウンロードCSVキャッシュ
└── logs/                        # トレードログ
```

### ブローカー抽象化
`config_loader.py` の `create_broker()` ファクトリ関数が `.env` の `DATA_SOURCE` に基づき
適切なクライアントを生成。両クライアントは同一インターフェースを実装:
- `connect()` / `get_historical_prices()` / `get_market_info()`
- `open_position()` / `close_position()` / `get_positions()`

## AI仙人のメソドロジー実装

### シグナル生成パイプライン
```
パターン検出 → AI確認（Gemini） → MTF確認 → インジケーター合流 → リスク検証 → 実行
```

### パターン認識（6パターン）
- **ダブルボトム（W型）**: 右肩上がりバリアント優先（2番目の底が高い）
- **ダブルトップ（M型）**: ベアリッシュリバーサル
- **逆ヘッドアンドショルダーズ**: 右肩上がりバリアント優先
- **ヘッドアンドショルダーズ**: ベアリッシュリバーサル
- **チャネルブレイクアウト**: ATR倍率ベースのブレイクアウト検出
- **移動平均クロスオーバー**: ゴールデンクロス/デスクロス（20/50 SMA）

### マルチタイムフレーム分析
| タイムフレーム | 用途 | OANDA粒度 |
|---------------|------|-----------|
| 日足 | トレンド方向判定 | DAY |
| 4時間足 | パターン検出（メイン） | HOUR_4 |
| 1時間足 | エントリータイミング | HOUR |

### AI ビジュアル分析（Google Gemini）
1. mplfinanceでローソク足チャート画像を生成（SMA/BB重畳）
2. Gemini 2.5 Flash にチャート画像を送信
3. JSON形式でパターン・方向・信頼度・価格レベルを返却
4. アルゴリズム検出結果とマージして最終シグナル生成

### テクニカル指標
- SMA: 20, 50, 100, 200
- EMA: 9, 21, 55, 200
- RSI (14), MACD (12/26/9), ボリンジャーバンド (20, 2σ)
- ATR (14), ADX (14)

### リスク管理
| パラメータ | 値 | 設定キー |
|-----------|-----|---------|
| 1トレードリスク | 口座の1% | `risk_per_trade_pct` |
| ストップロス | ATR × 2.0 | `stop_loss_atr_multiple` |
| テイクプロフィット | R:R 2:1 | `take_profit_rr_ratio` |
| 最小R:R | 0.5 | `min_risk_reward_ratio` |
| 最大ドローダウン | 10%で全停止 | `max_drawdown_pct` |
| デイリー損失上限 | 2% | `max_daily_loss_pct` |
| ウィークリー損失上限 | 5% | `max_weekly_loss_pct` |
| 最大同時ポジション | 3 | `max_open_positions` |
| トレーリングストップ | ATR × 1.5 | `trailing_stop_atr_multiple` |

### 適応的リスク制御（AI駆動）
トレード実績と市場状態に基づき、ポジションサイズを動的に調整:

| 要因 | 動作 | 設定キー |
|------|------|---------|
| 連敗スケーリング | 連敗ごとにリスク×0.75（最低25%まで） | `adaptive.loss_scale_factor` |
| ドローダウン防御 | DD 3%→×0.75、5%→×0.50、7%→×0.25 | `adaptive.drawdown_thresholds` |
| AI市場レジーム | Geminiがトレンド/レンジ/高ボラを判定し倍率設定 | AI自動判定 |
| 勝率モメンタム | 直近20トレードの勝率に応じて調整 | `adaptive.performance_window` |
| パターン無効化 | 勝率25%未満のパターンを自動スキップ | `adaptive.pattern_disable_win_rate` |
| 強制クールダウン | 5連敗で一時停止 | `adaptive.max_consecutive_losses` |

```
最終リスク = 基本リスク(1%) × 連敗係数 × DD係数 × レジーム係数 × 勝率係数
例: 1% × 0.75 × 0.50 × 0.8 × 0.85 = 0.255% （大幅縮小）
```

## セットアップ

```bash
# 依存ライブラリのインストール
pip install -r requirements.txt

# 環境変数の設定
cp .env.example .env
# .env を編集して APIキーを設定
```

### 必要なAPIキー
1. **Twelve Data API**（現在使用中）: https://twelvedata.com/register で無料登録
2. **Google Gemini API**: https://aistudio.google.com/ で取得（無料枠: 15 RPM / 100万トークン/日）
3. **OANDA API**（将来用）: ライブ口座開設後にAPIトークンを取得

### .env 設定例
```bash
DATA_SOURCE=twelvedata          # twelvedata または oanda
TWELVEDATA_API_KEY=your_key     # Twelve Data APIキー
GEMINI_API_KEY=your_key         # Gemini APIキー
TRADING_MODE=paper              # paper（ペーパートレード）

# メール通知（Gmail例）
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_app_password  # Googleアプリパスワード
EMAIL_FROM=your_email@gmail.com
EMAIL_TO=your_email@gmail.com
```

### ローカル常駐インストール

**macOS:**
```bash
chmod +x deploy/install_mac.sh
./deploy/install_mac.sh
nano ~/.ai-fx-bot/.env                              # APIキー設定
launchctl load ~/Library/LaunchAgents/com.ai-fx-bot.plist  # 起動
launchctl unload ~/Library/LaunchAgents/com.ai-fx-bot.plist  # 停止
tail -f ~/.ai-fx-bot/logs/trading.log                # ログ監視
```

**Linux:**
```bash
chmod +x deploy/install.sh
sudo ./deploy/install.sh
sudo nano /opt/ai-fx-bot/.env                        # APIキー設定
sudo systemctl start ai-fx-bot                       # 起動
sudo systemctl stop ai-fx-bot                        # 停止
journalctl -u ai-fx-bot -f                           # ログ監視
```

## 使い方

```bash
# ペーパートレードモード（デフォルト）
python main.py

# バックテスト（APIデータ: 約3ヶ月分）
python main.py --backtest
python main.py --backtest --instrument USD_JPY

# 長期バックテスト（yfinance: 最大2年分を自動DL）
python main.py --backtest --download --years 2
python main.py --backtest --instrument USD_JPY --download

# 外部CSVでバックテスト
python main.py --backtest --instrument USD_JPY --csv data/my_data.csv

# 特定ペアの分析
python main.py --analyze USD_JPY

# アカウントステータス確認
python main.py --status
```

### アクティブ通貨ペア
- USD_JPY, EUR_USD, GBP_JPY（`trading_config.yaml` の `active_pairs` で変更可能）

## テスト

```bash
# 全テスト実行（46テスト）
pytest tests/ -v

# 個別テスト
pytest tests/test_patterns.py -v      # パターン検出テスト
pytest tests/test_risk_manager.py -v  # リスク管理テスト
pytest tests/test_indicators.py -v    # テクニカル指標テスト
pytest tests/test_backtest.py -v      # バックテストテスト
```

## コーディング規約
- PEP 8 準拠
- 全関数に型ヒント
- パブリックなクラス・関数にdocstring
- トレード戦略のロジックは必ずテストを書く
- シークレット情報は `.env` に記載（コードに直書き禁止）

## バックテスト結果（Fixed vs Adaptive）

バックテストは `--backtest` で Fixed（従来）と Adaptive（適応的）を自動比較表示。

| 指標 | USD/JPY Fixed→Adaptive | EUR/USD Fixed→Adaptive | GBP/JPY Fixed→Adaptive |
|------|:---:|:---:|:---:|
| 勝率 | 52.5% → **67.7%** | 41.2% → **68.0%** | 52.8% → **73.8%** |
| PF | 2.13 → **5.70** | 1.11 → **5.34** | 1.81 → **4.76** |
| 最大DD | 43.4% → **8.3%** | 49.7% → **6.0%** | 60.2% → **6.8%** |
| シャープ | 3.47 → **6.19** | 0.57 → **6.98** | 2.61 → **6.34** |
| 最大連敗 | 32 → **11** | 42 → **14** | 35 → **20** |

## 既知の課題・今後の改善点
- `google-generativeai` パッケージの非推奨警告 → `google-genai` への移行が必要
- Discord/LINE通知機能は未設定（notifier.py は実装済み）
