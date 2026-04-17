# データ層

## テクニカル指標（`indicators.py`）

- SMA: 20, 50, 100, 200
- EMA: 9, 21, 55, 200
- RSI (14), MACD (12/26/9), ボリンジャーバンド (20, 2σ)
- ATR (14), ADX (14)

## データフェッチャー使い分け

| シナリオ | 使用クラス | ファイル |
| --- | --- | --- |
| リアルタイム取得（ペーパートレード） | `TwelveDataClient` / `OandaClient` | `src/brokers/` |
| バックテスト（APIデータ、~3ヶ月） | `MarketData` | `market_data.py` |
| 長期バックテスト（yfinance、最大2年） | `CSVLoader` | `csv_loader.py` |
| 外部CSVバックテスト | `CSVLoader` | `csv_loader.py` |
| センチメント取得 | `SentimentFetcher` | `sentiment_fetcher.py` |

## `csv_loader.py` vs `market_data.py` 責務分担

- **`market_data.py`**: ブローカーAPIからリアルタイムデータを取得・キャッシュ管理
- **`csv_loader.py`**: ファイルシステムまたはyfinanceからバッチデータをロード。`--download` フラグで自動DL
