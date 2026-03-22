"""移動平均クロス戦略。

短期MAが長期MAを上抜けたら買い（ゴールデンクロス）、
下抜けたら売り（デッドクロス）を発生させるシンプルな戦略。
"""

import pandas as pd

from src.data.processors import TechnicalIndicators


class MovingAverageCrossStrategy:
    """移動平均クロス戦略。

    Args:
        short_window: 短期MA期間（デフォルト20）
        long_window: 長期MA期間（デフォルト50）
    """

    def __init__(self, short_window: int = 20, long_window: int = 50) -> None:
        if short_window >= long_window:
            raise ValueError("short_window は long_window より小さい必要があります")
        self.short_window = short_window
        self.long_window = long_window
        self._indicators = TechnicalIndicators()

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """売買シグナルを生成する。

        Args:
            df: OHLCV DataFrame

        Returns:
            以下のカラムを追加した DataFrame:
                - SMA_short: 短期移動平均
                - SMA_long: 長期移動平均
                - signal: 1=買い, -1=売り, 0=ホールド
                - position: 現在のポジション方向（1 or 0）
        """
        result = df.copy()
        result["SMA_short"] = TechnicalIndicators.sma(df, self.short_window)
        result["SMA_long"] = TechnicalIndicators.sma(df, self.long_window)

        # 短期MAが長期MAより上にあるとき position=1、そうでなければ 0
        result["position"] = (result["SMA_short"] > result["SMA_long"]).astype(int)

        # ポジション変化点がシグナル（1=買い, -1=売り）
        result["signal"] = result["position"].diff().fillna(0).astype(int)

        return result.dropna(subset=["SMA_short", "SMA_long"])

    def backtest(self, df: pd.DataFrame) -> pd.DataFrame:
        """簡易バックテストを実行し、損益を計算する。

        Args:
            df: OHLCV DataFrame

        Returns:
            以下のカラムを追加した DataFrame:
                - returns: 日次リターン
                - strategy_returns: 戦略リターン
                - cumulative_returns: 累積リターン（バイアンドホールド）
                - cumulative_strategy: 累積戦略リターン
        """
        result = self.generate_signals(df)
        result["returns"] = result["Close"].pct_change()
        result["strategy_returns"] = result["returns"] * result["position"].shift(1)
        result["cumulative_returns"] = (1 + result["returns"]).cumprod()
        result["cumulative_strategy"] = (1 + result["strategy_returns"]).cumprod()
        return result

    def summary(self, df: pd.DataFrame) -> dict:
        """バックテスト結果のサマリーを返す。

        Args:
            df: backtest() の戻り値

        Returns:
            {
                total_return: 戦略の総リターン,
                buy_and_hold_return: バイアンドホールドの総リターン,
                num_trades: 取引回数,
                sharpe_ratio: シャープレシオ（年率換算）
            }
        """
        result = self.backtest(df)
        strategy_ret = result["strategy_returns"].dropna()

        total_return = result["cumulative_strategy"].iloc[-1] - 1
        bh_return = result["cumulative_returns"].iloc[-1] - 1
        num_trades = (result["signal"] != 0).sum()
        sharpe = (
            strategy_ret.mean() / strategy_ret.std() * (252 ** 0.5)
            if strategy_ret.std() != 0
            else 0.0
        )

        return {
            "total_return": round(float(total_return), 4),
            "buy_and_hold_return": round(float(bh_return), 4),
            "num_trades": int(num_trades),
            "sharpe_ratio": round(float(sharpe), 4),
        }
