"""市場データ取得モジュール。yfinance を使って OHLCV データを取得する。"""

from datetime import datetime
from typing import Optional

import pandas as pd
import yfinance as yf


class MarketDataFetcher:
    """Yahoo Finance から市場データを取得するクラス。"""

    def fetch(
        self,
        symbol: str,
        start: str,
        end: Optional[str] = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """OHLCV データを取得する。

        Args:
            symbol: ティッカーシンボル（例: "AAPL", "7203.T"）
            start: 取得開始日（"YYYY-MM-DD"）
            end: 取得終了日（"YYYY-MM-DD"）。省略時は本日。
            interval: 時間足（"1d", "1h", "5m" など）

        Returns:
            Open, High, Low, Close, Volume の DataFrame
        """
        if end is None:
            end = datetime.today().strftime("%Y-%m-%d")

        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start, end=end, interval=interval)

        if df.empty:
            raise ValueError(f"データが取得できませんでした: {symbol} ({start} 〜 {end})")

        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index.name = "Date"
        return df

    def fetch_multiple(
        self,
        symbols: list[str],
        start: str,
        end: Optional[str] = None,
        interval: str = "1d",
    ) -> dict[str, pd.DataFrame]:
        """複数銘柄のデータをまとめて取得する。

        Args:
            symbols: ティッカーシンボルのリスト
            start: 取得開始日
            end: 取得終了日
            interval: 時間足

        Returns:
            {symbol: DataFrame} の辞書
        """
        return {
            symbol: self.fetch(symbol, start, end, interval)
            for symbol in symbols
        }
