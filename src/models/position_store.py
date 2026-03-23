"""SQLite-based position persistence store.

ポジション情報をSQLiteに保存・復元する。
テキストログのパース依存を解消し、再起動時の状態復元を確実にする。
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger


class PositionStore:
    """SQLiteでポジションを永続管理するクラス。"""

    def __init__(self, db_path: str = "data/positions.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """テーブルを初期化する。"""
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    deal_id     TEXT PRIMARY KEY,
                    instrument  TEXT NOT NULL,
                    pair_name   TEXT NOT NULL,
                    direction   TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    size        REAL NOT NULL,
                    stop_loss   REAL,
                    take_profit REAL,
                    pattern     TEXT,
                    opened_at   TEXT NOT NULL,
                    extra       TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trade_history (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    deal_id     TEXT NOT NULL,
                    instrument  TEXT NOT NULL,
                    pair_name   TEXT NOT NULL,
                    direction   TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price  REAL,
                    size        REAL NOT NULL,
                    pnl         REAL,
                    pattern     TEXT,
                    opened_at   TEXT NOT NULL,
                    closed_at   TEXT
                )
            """)

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    # --------------------------------------------------
    # ポジション操作
    # --------------------------------------------------

    def save_position(self, trade: dict[str, Any]) -> None:
        """ポジションを保存する。"""
        try:
            extra = {k: v for k, v in trade.items()
                     if k not in ("deal_id", "instrument", "pair_name", "direction",
                                  "entry_price", "size", "stop_loss", "take_profit",
                                  "pattern", "opened_at")}
            with self._conn() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO positions
                    (deal_id, instrument, pair_name, direction, entry_price, size,
                     stop_loss, take_profit, pattern, opened_at, extra)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trade["deal_id"],
                    trade["instrument"],
                    trade.get("pair_name", trade["instrument"].replace("_", "/")),
                    trade["direction"],
                    trade["entry_price"],
                    trade["size"],
                    trade.get("stop_loss"),
                    trade.get("take_profit"),
                    trade.get("pattern"),
                    trade.get("opened_at", datetime.now().isoformat()),
                    json.dumps(extra) if extra else None,
                ))
            logger.debug(f"Position saved: {trade['deal_id']} {trade['instrument']}")
        except Exception as e:
            logger.error(f"Failed to save position: {e}")

    def update_position(self, deal_id: str, stop_loss: float | None = None,
                        take_profit: float | None = None) -> None:
        """SL/TPを更新する。"""
        try:
            with self._conn() as conn:
                if stop_loss is not None:
                    conn.execute("UPDATE positions SET stop_loss=? WHERE deal_id=?",
                                 (stop_loss, deal_id))
                if take_profit is not None:
                    conn.execute("UPDATE positions SET take_profit=? WHERE deal_id=?",
                                 (take_profit, deal_id))
        except Exception as e:
            logger.error(f"Failed to update position {deal_id}: {e}")

    def close_position(self, deal_id: str, exit_price: float, pnl: float) -> None:
        """ポジションをクローズして履歴に移動する。"""
        try:
            with self._conn() as conn:
                row = conn.execute(
                    "SELECT * FROM positions WHERE deal_id=?", (deal_id,)
                ).fetchone()
                if row is None:
                    logger.warning(f"Position not found in DB: {deal_id}")
                    return
                cols = [d[0] for d in conn.execute(
                    "SELECT * FROM positions WHERE deal_id=?", (deal_id,)
                ).description]
                pos = dict(zip(cols, row))

                conn.execute("""
                    INSERT INTO trade_history
                    (deal_id, instrument, pair_name, direction, entry_price, exit_price,
                     size, pnl, pattern, opened_at, closed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    pos["deal_id"], pos["instrument"], pos["pair_name"],
                    pos["direction"], pos["entry_price"], exit_price,
                    pos["size"], pnl, pos["pattern"], pos["opened_at"],
                    datetime.now().isoformat(),
                ))
                conn.execute("DELETE FROM positions WHERE deal_id=?", (deal_id,))
            logger.debug(f"Position closed in DB: {deal_id} pnl={pnl:+.0f}")
        except Exception as e:
            logger.error(f"Failed to close position in DB {deal_id}: {e}")

    def get_open_positions(self) -> list[dict[str, Any]]:
        """全オープンポジションを取得する。"""
        try:
            with self._conn() as conn:
                rows = conn.execute("SELECT * FROM positions").fetchall()
                if not rows:
                    return []
                cols = [d[0] for d in conn.execute("SELECT * FROM positions").description]
                result = []
                for row in rows:
                    pos = dict(zip(cols, row))
                    if pos.get("extra"):
                        pos.update(json.loads(pos["extra"]))
                    del pos["extra"]
                    result.append(pos)
                return result
        except Exception as e:
            logger.error(f"Failed to get open positions: {e}")
            return []

    def has_open_position(self, instrument: str) -> bool:
        """指定銘柄のオープンポジションが存在するか確認する。"""
        try:
            with self._conn() as conn:
                count = conn.execute(
                    "SELECT COUNT(*) FROM positions WHERE instrument=?", (instrument,)
                ).fetchone()[0]
                return count > 0
        except Exception as e:
            logger.error(f"Failed to check position for {instrument}: {e}")
            return False

    def get_next_deal_id(self) -> int:
        """次のPAPERトレードID番号を返す（全履歴含む最大値+1）。"""
        try:
            with self._conn() as conn:
                max_pos = conn.execute(
                    "SELECT MAX(CAST(REPLACE(deal_id,'PAPER-','') AS INTEGER)) FROM positions"
                ).fetchone()[0] or 0
                max_hist = conn.execute(
                    "SELECT MAX(CAST(REPLACE(deal_id,'PAPER-','') AS INTEGER)) FROM trade_history"
                ).fetchone()[0] or 0
                return max(max_pos, max_hist) + 1
        except Exception as e:
            logger.error(f"Failed to get next deal id: {e}")
            return 1

    def get_trade_history(self, limit: int = 50) -> list[dict[str, Any]]:
        """取引履歴を取得する。"""
        try:
            with self._conn() as conn:
                rows = conn.execute(
                    "SELECT * FROM trade_history ORDER BY closed_at DESC LIMIT ?",
                    (limit,)
                ).fetchall()
                if not rows:
                    return []
                cols = [d[0] for d in conn.execute(
                    "SELECT * FROM trade_history ORDER BY closed_at DESC LIMIT ?",
                    (limit,)
                ).description]
                return [dict(zip(cols, row)) for row in rows]
        except Exception as e:
            logger.error(f"Failed to get trade history: {e}")
            return []

    def get_daily_pnl(self) -> float:
        """本日の損益合計を返す。"""
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            with self._conn() as conn:
                result = conn.execute(
                    "SELECT SUM(pnl) FROM trade_history WHERE closed_at LIKE ?",
                    (f"{today}%",)
                ).fetchone()[0]
                return float(result or 0.0)
        except Exception as e:
            logger.error(f"Failed to get daily pnl: {e}")
            return 0.0
