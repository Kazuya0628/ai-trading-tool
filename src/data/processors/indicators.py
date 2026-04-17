"""テクニカル指標の計算モジュール。"""

import pandas as pd


class TechnicalIndicators:
    """OHLCV DataFrame にテクニカル指標を付与するクラス。"""

    @staticmethod
    def sma(df: pd.DataFrame, window: int, column: str = "Close") -> pd.Series:
        """単純移動平均（SMA）を計算する。

        Args:
            df: OHLCV DataFrame
            window: 期間
            column: 対象カラム

        Returns:
            SMA の Series
        """
        return df[column].rolling(window=window).mean()

    @staticmethod
    def ema(df: pd.DataFrame, window: int, column: str = "Close") -> pd.Series:
        """指数移動平均（EMA）を計算する。

        Args:
            df: OHLCV DataFrame
            window: 期間
            column: 対象カラム

        Returns:
            EMA の Series
        """
        return df[column].ewm(span=window, adjust=False).mean()

    @staticmethod
    def rsi(df: pd.DataFrame, window: int = 14, column: str = "Close") -> pd.Series:
        """RSI（相対力指数）を計算する。

        Args:
            df: OHLCV DataFrame
            window: 期間（デフォルト14）
            column: 対象カラム

        Returns:
            RSI の Series（0〜100）
        """
        delta = df[column].diff()
        gain = delta.clip(lower=0).rolling(window=window).mean()
        loss = (-delta.clip(upper=0)).rolling(window=window).mean()
        rs = gain / loss.replace(0, float("nan"))
        return 100 - (100 / (1 + rs))

    @staticmethod
    def bollinger_bands(
        df: pd.DataFrame, window: int = 20, num_std: float = 2.0, column: str = "Close"
    ) -> pd.DataFrame:
        """ボリンジャーバンドを計算する。

        Args:
            df: OHLCV DataFrame
            window: 期間（デフォルト20）
            num_std: 標準偏差の倍率（デフォルト2.0）
            column: 対象カラム

        Returns:
            upper, middle, lower カラムを持つ DataFrame
        """
        middle = df[column].rolling(window=window).mean()
        std = df[column].rolling(window=window).std()
        return pd.DataFrame(
            {
                "upper": middle + num_std * std,
                "middle": middle,
                "lower": middle - num_std * std,
            },
            index=df.index,
        )

    def add_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """よく使う指標をまとめて追加した DataFrame を返す。

        追加される列:
            SMA_20, SMA_50, EMA_20, RSI_14,
            BB_upper, BB_middle, BB_lower

        Args:
            df: OHLCV DataFrame

        Returns:
            指標列を追加した新しい DataFrame
        """
        result = df.copy()
        result["SMA_20"] = self.sma(df, 20)
        result["SMA_50"] = self.sma(df, 50)
        result["EMA_20"] = self.ema(df, 20)
        result["RSI_14"] = self.rsi(df)
        bb = self.bollinger_bands(df)
        result["BB_upper"] = bb["upper"]
        result["BB_middle"] = bb["middle"]
        result["BB_lower"] = bb["lower"]
        return result
