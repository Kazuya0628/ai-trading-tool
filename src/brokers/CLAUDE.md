# ブローカー層

## データソース

| ソース | 状態 | 用途 |
| --- | --- | --- |
| **Twelve Data** | 現在アクティブ | 無料枠（800リクエスト/日、8リクエスト/分） |
| **OANDA v20** | 将来切替用 | ライブ口座取得後に `.env` の `DATA_SOURCE=oanda` で切替 |

切替は `.env` の `DATA_SOURCE` を変更するだけ（コード変更不要）。

## ブローカー共通インターフェース（`base_broker.py`）

両クライアントが実装すべき6メソッド:

- `connect()` — 接続・認証
- `get_historical_prices(instrument, granularity, count)` — 価格履歴取得
- `get_market_info(instrument)` — 銘柄情報取得
- `open_position(instrument, units, side)` — ポジション開設
- `close_position(trade_id)` — ポジション決済
- `get_positions()` — 保有ポジション一覧

## ファクトリ

`src/utils/config_loader.py` の `create_broker()` が `.env` の `DATA_SOURCE` を読み取り、
`TwelveDataClient` または `OandaClient` のインスタンスを返す。
呼び出し側はブローカー種別を意識しない。

## Twelve Data 制限

- 無料枠: 800リクエスト/日、8リクエスト/分
- レート超過時は自動スリープで待機（`twelvedata_client.py` 内で処理）

## OANDA v20（将来用）

ライブ口座開設後にAPIトークンを取得し `.env` の `OANDA_API_KEY` と `OANDA_ACCOUNT_ID` に設定。
