# AI FX Trading Tool - AI仙人メソドロジー自動売買システム

AI仙人のメソドロジーに基づくAI搭載FX分析・ペーパートレードツール。
チャートパターン検出 + Gemini AIビジュアル分析によるシグナル生成・ペーパートレードを実行します。

## サブディレクトリのコンテキスト

詳細情報は各サブディレクトリの `CLAUDE.md` に分散しています:

| ディレクトリ | CLAUDE.md の内容 |
| --- | --- |
| `src/brokers/` | データソース、ブローカーインターフェース、API切替方法 |
| `src/strategies/` | AI仙人メソドロジー、シグナルパイプライン、6パターン、MTF、Gemini分析 |
| `src/models/` | リスク管理パラメータ、適応的リスク制御、バックテスト結果 |
| `src/data/` | テクニカル指標、データフェッチャー使い分け |
| `deploy/` | APIキー、.env設定、macOS/Linuxインストール手順 |
| `tests/` | テスト実行コマンド、テストファイル一覧 |

## ブローカー切替

`.env` の `DATA_SOURCE` を `twelvedata` または `oanda` に変更するだけ（コード変更不要）。

## 使い方

```bash
python main.py                                        # ペーパートレード
python main.py --backtest                             # バックテスト（Fixed vs Adaptive自動比較）
python main.py --backtest --instrument USD_JPY        # 特定ペアのバックテスト
python main.py --backtest --download --years 2        # 長期バックテスト（yfinance）
python main.py --backtest --instrument USD_JPY --csv data/my_data.csv  # CSVバックテスト
python main.py --analyze USD_JPY                      # 特定ペアの分析
python main.py --status                               # アカウントステータス確認
python src/dashboard.py                               # Webダッシュボード（既定: http://localhost:5000 / 使用中なら5001-5020で自動フォールバック）
```

### アクティブ通貨ペア

- USD_JPY, EUR_USD, GBP_JPY（`config/trading_config.yaml` の `active_pairs` で変更可能）

## コーディング規約
- PEP 8 準拠
- 全関数に型ヒント
- パブリックなクラス・関数にdocstring
- トレード戦略のロジックは必ずテストを書く
- シークレット情報は `.env` に記載（コードに直書き禁止）

## 既知の課題
- `google-generativeai` パッケージの非推奨警告 → `google-genai` への移行が必要
- Discord/LINE通知機能は未設定（`src/utils/notifier.py` は実装済み）
