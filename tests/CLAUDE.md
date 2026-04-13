# テスト

## テスト実行コマンド

```bash
pytest tests/ -v                       # 全テスト実行（46テスト）
pytest tests/test_patterns.py -v       # パターン検出テスト
pytest tests/test_risk_manager.py -v   # リスク管理テスト
pytest tests/test_indicators.py -v     # テクニカル指標テスト
pytest tests/test_backtest.py -v       # バックテストテスト
pytest tests/test_ma_cross.py -v       # MAクロスオーバーテスト
```

## テストファイル一覧

| ファイル | カバー領域 |
| --- | --- |
| `test_patterns.py` | `pattern_detector.py` — 6チャートパターン検出 |
| `test_risk_manager.py` | `risk_manager.py` — リスク計算・適応的スケーリング |
| `test_indicators.py` | `indicators.py` — SMA/EMA/RSI/MACD/ATR/ADX計算 |
| `test_backtest.py` | `backtest_engine.py` — Fixed/Adaptive バックテスト |
| `test_ma_cross.py` | `ma_cross.py` — MAクロスオーバー戦略 |

## 規約

トレード戦略のロジック（`src/strategies/`、`src/models/`）は**必ずテストを書く**。
新パターンや新リスクルールを追加した場合は対応テストを同一PRに含めること。
