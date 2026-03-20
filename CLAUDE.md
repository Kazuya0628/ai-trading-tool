# AiTradeTool - Claude Code 指示書

## プロジェクト概要
Pythonで構築するAI搭載のトレーディングツール。

## フォルダ構成

```
AiTradeTool/
├── CLAUDE.md              # このファイル（Claude Code への指示）
├── .gitignore
├── .env.example           # 環境変数のテンプレート（.env は絶対にコミットしない）
├── requirements.txt       # Python依存ライブラリ
├── src/                   # ソースコード
│   ├── data/              # 市場データの取得・加工
│   ├── models/            # AI/MLモデル（価格予測など）
│   ├── strategies/        # トレード戦略の実装
│   ├── brokers/           # ブローカーAPIとの連携
│   └── utils/             # 共通ユーティリティ
├── tests/                 # ユニット・統合テスト
├── notebooks/             # 分析・検証用 Jupyter ノートブック
├── docs/                  # 参考資料・仕様書
│   ├── api/               # ブローカー・データプロバイダーのAPI仕様
│   └── research/          # 参考論文・調査資料
└── data/
    ├── raw/               # 取得した生データ
    └── processed/         # 加工済み・特徴量データ
```

## 開発環境のセットアップ

```bash
# 仮想環境の作成と有効化
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 依存ライブラリのインストール
pip install -r requirements.txt

# 環境変数の設定
cp .env.example .env
# .env を編集して APIキーを設定
```

## よく使うコマンド

```bash
# テストの実行
pytest tests/

# Jupyter ノートブックの起動
jupyter notebook notebooks/
```

## コーディング規約
- PEP 8 に従う
- 全ての関数に型ヒントを付ける
- パブリックなクラス・関数にはdocstringを書く
- トレード戦略のロジックはテストを必ず書く
- シークレット情報は `.env` に記載し、コードに直書きしない
