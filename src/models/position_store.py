"""SQLite-based position persistence store (DESIGN-001 v1.3).

ポジション情報をSQLiteに保存・復元する。
10テーブルスキーマ対応版。既存DBからの自動マイグレーション機能付き。
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

# ---------------------------------------------------------------------------
# DDL definitions for all 10 tables (DESIGN-001 Appendix A)
# ---------------------------------------------------------------------------

_DDL_POSITIONS = """
CREATE TABLE IF NOT EXISTS positions (
    deal_id         TEXT PRIMARY KEY,
    position_id     TEXT,
    broker_trade_id TEXT,
    signal_id       TEXT,
    instrument      TEXT NOT NULL,
    pair_name       TEXT NOT NULL,
    direction       TEXT NOT NULL,
    entry_price     REAL NOT NULL,
    exit_price      REAL,
    size            REAL NOT NULL,
    stop_loss       REAL,
    take_profit     REAL,
    pnl             REAL,
    pattern         TEXT,
    confidence      REAL,
    is_fallback     INTEGER DEFAULT 0,
    exit_reason     TEXT,
    opened_at       TEXT NOT NULL,
    closed_at       TEXT,
    extra           TEXT
)
"""

_DDL_TRADE_HISTORY = """
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
"""

_DDL_SIGNALS = """
CREATE TABLE IF NOT EXISTS signals (
    signal_id       TEXT PRIMARY KEY,
    instrument      TEXT NOT NULL,
    direction       TEXT NOT NULL,
    pattern         TEXT,
    confidence      REAL,
    is_fallback     INTEGER DEFAULT 0,
    timeframe       TEXT,
    source          TEXT,
    entry_suggested REAL,
    sl_suggested    REAL,
    tp_suggested    REAL,
    created_at      TEXT NOT NULL
)
"""

_DDL_AI_VOTES = """
CREATE TABLE IF NOT EXISTS ai_votes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id   TEXT NOT NULL,
    model_name  TEXT NOT NULL,
    direction   TEXT,
    confidence  REAL,
    reasoning   TEXT,
    latency_ms  INTEGER,
    created_at  TEXT NOT NULL,
    FOREIGN KEY (signal_id) REFERENCES signals(signal_id)
)
"""

_DDL_ORDERS = """
CREATE TABLE IF NOT EXISTS orders (
    order_id        TEXT PRIMARY KEY,
    signal_id       TEXT,
    instrument      TEXT NOT NULL,
    direction       TEXT NOT NULL,
    order_type      TEXT NOT NULL DEFAULT 'market',
    requested_price REAL,
    filled_price    REAL,
    size            REAL NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending',
    created_at      TEXT NOT NULL,
    filled_at       TEXT,
    FOREIGN KEY (signal_id) REFERENCES signals(signal_id)
)
"""

_DDL_TRADES = """
CREATE TABLE IF NOT EXISTS trades (
    trade_id    TEXT PRIMARY KEY,
    order_id    TEXT,
    deal_id     TEXT,
    instrument  TEXT NOT NULL,
    direction   TEXT NOT NULL,
    entry_price REAL NOT NULL,
    exit_price  REAL,
    size        REAL NOT NULL,
    pnl         REAL,
    fees        REAL DEFAULT 0,
    opened_at   TEXT NOT NULL,
    closed_at   TEXT,
    FOREIGN KEY (order_id) REFERENCES orders(order_id),
    FOREIGN KEY (deal_id) REFERENCES positions(deal_id)
)
"""

_DDL_ACCOUNT_SNAPSHOTS = """
CREATE TABLE IF NOT EXISTS account_snapshots (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    balance         REAL NOT NULL,
    equity          REAL NOT NULL,
    margin_used     REAL DEFAULT 0,
    open_positions  INTEGER DEFAULT 0,
    daily_pnl       REAL DEFAULT 0,
    drawdown_pct    REAL DEFAULT 0,
    snapshot_at     TEXT NOT NULL
)
"""

_DDL_API_USAGE_DAILY = """
CREATE TABLE IF NOT EXISTS api_usage_daily (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    date            TEXT NOT NULL,
    provider        TEXT NOT NULL,
    requests_count  INTEGER DEFAULT 0,
    tokens_in       INTEGER DEFAULT 0,
    tokens_out      INTEGER DEFAULT 0,
    cost_usd        REAL DEFAULT 0,
    UNIQUE(date, provider)
)
"""

_DDL_SYSTEM_STATE = """
CREATE TABLE IF NOT EXISTS system_state (
    key         TEXT PRIMARY KEY,
    value       TEXT,
    updated_at  TEXT NOT NULL
)
"""

_DDL_PATTERN_STATS = """
CREATE TABLE IF NOT EXISTS pattern_stats (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    pattern     TEXT NOT NULL,
    instrument  TEXT NOT NULL,
    wins        INTEGER DEFAULT 0,
    losses      INTEGER DEFAULT 0,
    total_pnl   REAL DEFAULT 0,
    avg_rr      REAL DEFAULT 0,
    updated_at  TEXT NOT NULL,
    UNIQUE(pattern, instrument)
)
"""

_DDL_EXECUTION_EVENTS = """
CREATE TABLE IF NOT EXISTS execution_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type  TEXT NOT NULL,
    deal_id     TEXT,
    detail      TEXT,
    created_at  TEXT NOT NULL
)
"""

_ALL_DDL = [
    _DDL_POSITIONS,
    _DDL_TRADE_HISTORY,
    _DDL_SIGNALS,
    _DDL_AI_VOTES,
    _DDL_ORDERS,
    _DDL_TRADES,
    _DDL_ACCOUNT_SNAPSHOTS,
    _DDL_API_USAGE_DAILY,
    _DDL_SYSTEM_STATE,
    _DDL_PATTERN_STATS,
    _DDL_EXECUTION_EVENTS,
]

# Columns that may need to be added to a pre-existing positions table
_POSITIONS_NEW_COLUMNS: list[tuple[str, str]] = [
    ("position_id", "TEXT"),
    ("broker_trade_id", "TEXT"),
    ("signal_id", "TEXT"),
    ("exit_price", "REAL"),
    ("pnl", "REAL"),
    ("exit_reason", "TEXT"),
    ("confidence", "REAL"),
    ("is_fallback", "INTEGER DEFAULT 0"),
    ("closed_at", "TEXT"),
]


class PositionStore:
    """SQLiteでポジション・シグナル・システム状態を永続管理するクラス。

    DESIGN-001 v1.3 準拠。既存DBとの後方互換性を維持しつつ、
    10テーブルスキーマへ自動マイグレーションする。
    """

    def __init__(self, db_path: str = "data/positions.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ==================================================================
    # Internal: DB connection and initialization
    # ==================================================================

    def _conn(self) -> sqlite3.Connection:
        """SQLite接続を返す。PRAGMAはここで設定する。"""
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        return conn

    def _init_db(self) -> None:
        """全テーブルを作成し、既存テーブルのマイグレーションを実行する。"""
        with self._conn() as conn:
            for ddl in _ALL_DDL:
                conn.execute(ddl)
            self._migrate_schema(conn)

    def _migrate_schema(self, conn: sqlite3.Connection) -> None:
        """既存の positions テーブルに不足カラムを追加する。

        ALTER TABLE ADD COLUMN は SQLite でカラムが既に存在する場合にエラーを
        返すため、各カラムを try/except で個別に追加する。
        """
        for col_name, col_type in _POSITIONS_NEW_COLUMNS:
            try:
                conn.execute(
                    f"ALTER TABLE positions ADD COLUMN {col_name} {col_type}"
                )
                logger.info(f"Migrated positions table: added column {col_name}")
            except sqlite3.OperationalError:
                # カラムが既に存在する場合は無視
                pass

    # ==================================================================
    # Position operations (backward-compatible public API)
    # ==================================================================

    def save_position(self, trade: dict[str, Any]) -> None:
        """ポジションを保存する。

        deal_id を position_id にもマッピングする。
        """
        try:
            known_keys = {
                "deal_id", "position_id", "broker_trade_id", "signal_id",
                "instrument", "pair_name", "direction", "entry_price",
                "exit_price", "size", "stop_loss", "take_profit", "pnl",
                "pattern", "confidence", "is_fallback", "exit_reason",
                "opened_at", "closed_at",
            }
            extra = {k: v for k, v in trade.items() if k not in known_keys}

            deal_id = trade["deal_id"]
            opened_at = trade.get("opened_at", datetime.now().isoformat())

            with self._conn() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO positions
                    (deal_id, position_id, broker_trade_id, signal_id,
                     instrument, pair_name, direction, entry_price, size,
                     stop_loss, take_profit, pattern, confidence, is_fallback,
                     opened_at, extra)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        deal_id,
                        trade.get("position_id", deal_id),
                        trade.get("broker_trade_id"),
                        trade.get("signal_id"),
                        trade["instrument"],
                        trade.get("pair_name", trade["instrument"].replace("_", "/")),
                        trade["direction"],
                        trade["entry_price"],
                        trade["size"],
                        trade.get("stop_loss"),
                        trade.get("take_profit"),
                        trade.get("pattern"),
                        trade.get("confidence"),
                        1 if trade.get("is_fallback") else 0,
                        opened_at,
                        json.dumps(extra) if extra else None,
                    ),
                )
            logger.debug(f"Position saved: {deal_id} {trade['instrument']}")
        except Exception as e:
            logger.error(f"Failed to save position: {e}")

    def update_position(
        self,
        deal_id: str,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> None:
        """SL/TPを更新する。"""
        try:
            with self._conn() as conn:
                if stop_loss is not None:
                    conn.execute(
                        "UPDATE positions SET stop_loss=? WHERE deal_id=? AND closed_at IS NULL",
                        (stop_loss, deal_id),
                    )
                if take_profit is not None:
                    conn.execute(
                        "UPDATE positions SET take_profit=? WHERE deal_id=? AND closed_at IS NULL",
                        (take_profit, deal_id),
                    )
        except Exception as e:
            logger.error(f"Failed to update position {deal_id}: {e}")

    def close_position(
        self,
        deal_id: str,
        exit_price: float,
        pnl: float,
        exit_reason: str = "manual",
    ) -> None:
        """ポジションをクローズする。

        positions テーブルの行を UPDATE で閉じ、後方互換のため
        trade_history にも書き込む（移行期間中）。
        """
        try:
            closed_at = datetime.now().isoformat()
            with self._conn() as conn:
                row = conn.execute(
                    "SELECT * FROM positions WHERE deal_id=?", (deal_id,)
                ).fetchone()
                if row is None:
                    logger.warning(f"Position not found in DB: {deal_id}")
                    return

                cols = [
                    d[0]
                    for d in conn.execute(
                        "SELECT * FROM positions WHERE deal_id=?", (deal_id,)
                    ).description
                ]
                pos = dict(zip(cols, row))

                # UPDATE the positions row (new behavior)
                conn.execute(
                    """
                    UPDATE positions
                    SET exit_price=?, pnl=?, exit_reason=?, closed_at=?
                    WHERE deal_id=?
                    """,
                    (exit_price, pnl, exit_reason, closed_at, deal_id),
                )

                # Write to trade_history for backward compatibility (transition)
                conn.execute(
                    """
                    INSERT INTO trade_history
                    (deal_id, instrument, pair_name, direction, entry_price,
                     exit_price, size, pnl, pattern, opened_at, closed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        pos["deal_id"],
                        pos["instrument"],
                        pos["pair_name"],
                        pos["direction"],
                        pos["entry_price"],
                        exit_price,
                        pos["size"],
                        pnl,
                        pos["pattern"],
                        pos["opened_at"],
                        closed_at,
                    ),
                )

            logger.debug(f"Position closed in DB: {deal_id} pnl={pnl:+.0f}")
        except Exception as e:
            logger.error(f"Failed to close position in DB {deal_id}: {e}")

    def get_open_positions(self) -> list[dict[str, Any]]:
        """全オープンポジション（closed_at IS NULL）を取得する。"""
        try:
            with self._conn() as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT * FROM positions WHERE closed_at IS NULL"
                ).fetchall()
                if not rows:
                    return []
                result: list[dict[str, Any]] = []
                for row in rows:
                    pos = dict(row)
                    if pos.get("extra"):
                        pos.update(json.loads(pos["extra"]))
                    pos.pop("extra", None)
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
                    "SELECT COUNT(*) FROM positions WHERE instrument=? AND closed_at IS NULL",
                    (instrument,),
                ).fetchone()[0]
                return count > 0
        except Exception as e:
            logger.error(f"Failed to check position for {instrument}: {e}")
            return False

    def get_position_by_deal_id(self, deal_id: str) -> dict[str, Any] | None:
        """指定deal_idのポジション情報（AIメタデータ含む）を取得する。

        OANDAから取得したポジションにAIメタデータ（pattern, confidence, signal_id等）を
        付加するために使用する。

        Args:
            deal_id: ポジションのdeal ID（例: 'OANDA-123'）。

        Returns:
            ポジション情報dict、見つからない場合はNone。
        """
        try:
            with self._conn() as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    "SELECT * FROM positions WHERE deal_id=?",
                    (deal_id,),
                ).fetchone()
                if not row:
                    return None
                pos = dict(row)
                if pos.get("extra"):
                    pos.update(json.loads(pos["extra"]))
                pos.pop("extra", None)
                return pos
        except Exception as e:
            logger.error(f"Failed to get position by deal_id {deal_id}: {e}")
            return None

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
        """取引履歴を取得する。

        positions テーブルの closed_at IS NOT NULL を優先し、
        不足分は trade_history テーブルからフォールバックする。
        """
        try:
            with self._conn() as conn:
                conn.row_factory = sqlite3.Row
                # Primary source: closed positions in positions table
                rows = conn.execute(
                    """
                    SELECT deal_id, instrument, pair_name, direction,
                           entry_price, exit_price, size, pnl, pattern,
                           opened_at, closed_at
                    FROM positions
                    WHERE closed_at IS NOT NULL
                    ORDER BY closed_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()

                result = [dict(r) for r in rows]
                seen_deal_ids = {r["deal_id"] for r in result}

                # Fallback: fill from trade_history if we have fewer than limit
                remaining = limit - len(result)
                if remaining > 0:
                    placeholders = ",".join("?" for _ in seen_deal_ids) if seen_deal_ids else "''"
                    if seen_deal_ids:
                        query = f"""
                            SELECT * FROM trade_history
                            WHERE deal_id NOT IN ({placeholders})
                            ORDER BY closed_at DESC
                            LIMIT ?
                        """
                        params: list[Any] = list(seen_deal_ids) + [remaining]
                    else:
                        query = """
                            SELECT * FROM trade_history
                            ORDER BY closed_at DESC
                            LIMIT ?
                        """
                        params = [remaining]

                    legacy_rows = conn.execute(query, params).fetchall()
                    result.extend(dict(r) for r in legacy_rows)

                # Re-sort combined results by closed_at descending
                result.sort(
                    key=lambda r: r.get("closed_at") or "", reverse=True
                )
                return result[:limit]
        except Exception as e:
            logger.error(f"Failed to get trade history: {e}")
            return []

    def get_daily_pnl(self) -> float:
        """本日の損益合計を返す。"""
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            with self._conn() as conn:
                # Primary: from positions table
                result = conn.execute(
                    "SELECT SUM(pnl) FROM positions WHERE closed_at IS NOT NULL AND closed_at LIKE ?",
                    (f"{today}%",),
                ).fetchone()[0]
                pnl_positions = float(result or 0.0)

                # Also check trade_history for records not yet in positions
                result2 = conn.execute(
                    """
                    SELECT SUM(pnl) FROM trade_history
                    WHERE closed_at LIKE ?
                    AND deal_id NOT IN (
                        SELECT deal_id FROM positions WHERE closed_at IS NOT NULL
                    )
                    """,
                    (f"{today}%",),
                ).fetchone()[0]
                pnl_legacy = float(result2 or 0.0)

                return pnl_positions + pnl_legacy
        except Exception as e:
            logger.error(f"Failed to get daily pnl: {e}")
            return 0.0

    # ==================================================================
    # Signal operations (new in DESIGN-001)
    # ==================================================================

    def save_signal(self, signal: dict[str, Any]) -> None:
        """シグナルを保存する。"""
        try:
            with self._conn() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO signals
                    (signal_id, instrument, direction, pattern, confidence,
                     is_fallback, timeframe, source, entry_suggested,
                     sl_suggested, tp_suggested, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        signal["signal_id"],
                        signal["instrument"],
                        signal["direction"],
                        signal.get("pattern"),
                        signal.get("confidence"),
                        1 if signal.get("is_fallback") else 0,
                        signal.get("timeframe"),
                        signal.get("source"),
                        signal.get("entry_suggested"),
                        signal.get("sl_suggested"),
                        signal.get("tp_suggested"),
                        signal.get("created_at", datetime.now().isoformat()),
                    ),
                )
            logger.debug(f"Signal saved: {signal['signal_id']}")
        except Exception as e:
            logger.error(f"Failed to save signal: {e}")

    def save_ai_vote(self, vote: dict[str, Any]) -> None:
        """AI投票結果を保存する。"""
        try:
            with self._conn() as conn:
                conn.execute(
                    """
                    INSERT INTO ai_votes
                    (signal_id, model_name, direction, confidence,
                     reasoning, latency_ms, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        vote["signal_id"],
                        vote["model_name"],
                        vote.get("direction"),
                        vote.get("confidence"),
                        vote.get("reasoning"),
                        vote.get("latency_ms"),
                        vote.get("created_at", datetime.now().isoformat()),
                    ),
                )
            logger.debug(f"AI vote saved: {vote['model_name']} for {vote['signal_id']}")
        except Exception as e:
            logger.error(f"Failed to save AI vote: {e}")

    def save_order(self, order: dict[str, Any]) -> None:
        """注文を保存する。"""
        try:
            with self._conn() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO orders
                    (order_id, signal_id, instrument, direction, order_type,
                     requested_price, filled_price, size, status,
                     created_at, filled_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        order["order_id"],
                        order.get("signal_id"),
                        order["instrument"],
                        order["direction"],
                        order.get("order_type", "market"),
                        order.get("requested_price"),
                        order.get("filled_price"),
                        order["size"],
                        order.get("status", "pending"),
                        order.get("created_at", datetime.now().isoformat()),
                        order.get("filled_at"),
                    ),
                )
            logger.debug(f"Order saved: {order['order_id']}")
        except Exception as e:
            logger.error(f"Failed to save order: {e}")

    # ==================================================================
    # Account snapshots
    # ==================================================================

    def save_account_snapshot(self, snapshot: dict[str, Any]) -> None:
        """口座スナップショットを保存する。"""
        try:
            with self._conn() as conn:
                conn.execute(
                    """
                    INSERT INTO account_snapshots
                    (balance, equity, margin_used, open_positions,
                     daily_pnl, drawdown_pct, snapshot_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        snapshot["balance"],
                        snapshot["equity"],
                        snapshot.get("margin_used", 0),
                        snapshot.get("open_positions", 0),
                        snapshot.get("daily_pnl", 0),
                        snapshot.get("drawdown_pct", 0),
                        snapshot.get("snapshot_at", datetime.now().isoformat()),
                    ),
                )
            logger.debug("Account snapshot saved")
        except Exception as e:
            logger.error(f"Failed to save account snapshot: {e}")

    # ==================================================================
    # System state (key-value store)
    # ==================================================================

    def get_system_state(self, key: str) -> str | None:
        """システム状態のキーに対応する値を取得する。"""
        try:
            with self._conn() as conn:
                row = conn.execute(
                    "SELECT value FROM system_state WHERE key=?", (key,)
                ).fetchone()
                return row[0] if row else None
        except Exception as e:
            logger.error(f"Failed to get system state '{key}': {e}")
            return None

    def set_system_state(self, key: str, value: str) -> None:
        """システム状態を保存/更新する。"""
        try:
            with self._conn() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO system_state (key, value, updated_at)
                    VALUES (?, ?, ?)
                    """,
                    (key, value, datetime.now().isoformat()),
                )
            logger.debug(f"System state set: {key}")
        except Exception as e:
            logger.error(f"Failed to set system state '{key}': {e}")

    # ==================================================================
    # Execution events
    # ==================================================================

    def log_execution_event(
        self,
        event_type: str,
        deal_id: str | None = None,
        detail: str | None = None,
    ) -> None:
        """実行イベントをログに記録する。"""
        try:
            with self._conn() as conn:
                conn.execute(
                    """
                    INSERT INTO execution_events
                    (event_type, deal_id, detail, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (event_type, deal_id, detail, datetime.now().isoformat()),
                )
            logger.debug(f"Execution event logged: {event_type}")
        except Exception as e:
            logger.error(f"Failed to log execution event: {e}")

    # ==================================================================
    # Pattern statistics
    # ==================================================================

    def update_pattern_stats(
        self,
        pattern: str,
        instrument: str,
        won: bool,
        pnl: float,
        risk_reward: float = 0.0,
    ) -> None:
        """パターン統計を更新する（UPSERT）。"""
        try:
            now = datetime.now().isoformat()
            with self._conn() as conn:
                existing = conn.execute(
                    "SELECT wins, losses, total_pnl, avg_rr FROM pattern_stats WHERE pattern=? AND instrument=?",
                    (pattern, instrument),
                ).fetchone()

                if existing is None:
                    wins = 1 if won else 0
                    losses = 0 if won else 1
                    conn.execute(
                        """
                        INSERT INTO pattern_stats
                        (pattern, instrument, wins, losses, total_pnl, avg_rr, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (pattern, instrument, wins, losses, pnl, risk_reward, now),
                    )
                else:
                    old_wins, old_losses, old_pnl, old_avg_rr = existing
                    new_wins = old_wins + (1 if won else 0)
                    new_losses = old_losses + (0 if won else 1)
                    new_total_pnl = old_pnl + pnl
                    total_trades = new_wins + new_losses
                    new_avg_rr = (
                        (old_avg_rr * (total_trades - 1) + risk_reward) / total_trades
                        if total_trades > 0
                        else 0.0
                    )
                    conn.execute(
                        """
                        UPDATE pattern_stats
                        SET wins=?, losses=?, total_pnl=?, avg_rr=?, updated_at=?
                        WHERE pattern=? AND instrument=?
                        """,
                        (new_wins, new_losses, new_total_pnl, new_avg_rr, now, pattern, instrument),
                    )
            logger.debug(f"Pattern stats updated: {pattern}/{instrument}")
        except Exception as e:
            logger.error(f"Failed to update pattern stats: {e}")
