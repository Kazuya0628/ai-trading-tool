# AI FX Trading Tool - IG Securities Automated Trading

## プロジェクト概要
AI仙人のメソドロジーに基づくAI搭載FX自動売買ツール。
IG証券 REST APIを使用して自動トレードを実行します。

## アーキテクチャ

```
AiTradeTool/
├── main.py                    # エントリーポイント（CLI）
├── config/
│   └── trading_config.yaml    # トレード設定ファイル
├── src/
│   ├── trading_bot.py         # メイン自動売買エンジン
│   ├── brokers/
│   │   └── ig_client.py       # IG証券 API クライアント
│   ├── data/
│   │   ├── market_data.py     # マーケットデータ取得・管理
│   │   └── indicators.py      # テクニカル指標計算
│   ├── models/
│   │   ├── risk_manager.py    # リスク管理・ポジションサイジング
│   │   └── backtest_engine.py # バックテストエンジン
│   ├── strategies/
│   │   ├── pattern_detector.py # チャートパターン検出
│   │   ├── ai_analyzer.py      # Gemini AI チャート分析
│   │   └── strategy_engine.py  # マルチ戦略エンジン
│   └── utils/
│       ├── config_loader.py   # 設定読み込み
│       ├── logger.py          # ログ設定
│       └── notifier.py        # Discord/LINE通知
├── tests/                     # テストスイート（46テスト）
├── data/                      # マーケットデータ保存
└── logs/                      # トレードログ
```

## AI仙人のメソドロジー実装

### パターン認識
- **ダブルボトム（W型）**: 右肩上がりバリアント優先
- **ダブルトップ（M型）**: ベアリッシュリバーサル
- **逆ヘッドアンドショルダーズ**: 右肩上がりバリアント優先
- **ヘッドアンドショルダーズ**: ベアリッシュリバーサル
- **チャネルブレイクアウト**: 基本戦略
- **移動平均クロスオーバー**: ゴールデンクロス/デスクロス

### マルチタイムフレーム分析
- 日足: トレンド方向判定
- 4時間足: パターン検出（メイン）
- 1時間足: エントリータイミング

### AI ビジュアル分析（Gemini）
1. チャート画像生成
2. Gemini AIによるパターン認識
3. バイナリシグナル（1=トレード, 0=ノーシグナル）
4. 実行判定

### リスク管理
- 1トレードあたりリスク: 口座の1%
- ATRベースのストップロス
- 2:1のリスクリワード比
- 最大ドローダウン: 10%で全停止
- デイリー損失上限: 2%
- トレーリングストップ対応

## セットアップ

```bash
# 依存ライブラリのインストール
pip install -r requirements.txt

# 環境変数の設定
cp .env.example .env
# .env を編集して APIキーを設定
```

### 必要なAPIキー
1. **IG証券 API**: https://labs.ig.com/ でAPIキーを取得
2. **Google Gemini API** (オプション): AI ビジュアル分析用

## 使い方

```bash
# ペーパートレードモード（デフォルト）
python main.py

# ライブトレード（注意: 実際の資金を使用）
python main.py --live

# バックテスト実行
python main.py --backtest
python main.py --backtest --epic CS.D.USDJPY.TODAY.IP

# 特定ペアの分析
python main.py --analyze CS.D.USDJPY.TODAY.IP

# アカウントステータス確認
python main.py --status
```

## コマンド

```bash
# テスト実行
pytest tests/ -v

# 特定テストの実行
pytest tests/test_patterns.py -v
pytest tests/test_risk_manager.py -v
pytest tests/test_indicators.py -v
pytest tests/test_backtest.py -v
```

## コーディング規約
- PEP 8 準拠
- 全関数に型ヒント
- パブリックなクラス・関数にdocstring
- トレード戦略のロジックは必ずテストを書く
- シークレット情報は `.env` に記載（コードに直書き禁止）
