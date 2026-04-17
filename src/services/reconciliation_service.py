"""Reconciliation Service — OANDA/DB consistency verification.

Responsibilities:
- Startup reconciliation (OANDA = truth source)
- Periodic consistency checks
- Gap detection and correction
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from loguru import logger


class ReconciliationService:
    """Reconciles OANDA open trades with local SQLite position store."""

    def __init__(self, broker: Any, position_store: Any) -> None:
        self.broker = broker
        self.position_store = position_store

    def reconcile_on_startup(self) -> dict[str, Any]:
        """Run full reconciliation at startup.

        Flow:
        1. Read open positions from SQLite
        2. Fetch open trades from OANDA
        3. OANDA is truth source — diff compare
        4. Fill gaps in DB from OANDA
        5. Mark DB-only positions as closed (reconciled)
        6. Log all discrepancies

        Returns:
            Reconciliation report.
        """
        report: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "db_positions": 0,
            "broker_positions": 0,
            "matched": 0,
            "db_only_closed": 0,
            "broker_only_added": 0,
            "errors": [],
        }

        try:
            # 1. DB open positions
            db_positions = self.position_store.get_open_positions()
            report["db_positions"] = len(db_positions)
            db_by_deal = {p.get("deal_id", ""): p for p in db_positions}

            # 2. Broker open trades (OANDA returns DataFrame)
            broker_df = self.broker.get_open_positions()
            if hasattr(broker_df, 'empty') and not broker_df.empty:
                broker_trades = broker_df.to_dict('records')
                # Normalize OANDA column names to internal names
                for t in broker_trades:
                    if "dealId" in t and "deal_id" not in t:
                        t["deal_id"] = t["dealId"]
                    if "epic" in t and "instrument" not in t:
                        t["instrument"] = t["epic"]
                    if "level" in t and "entry_price" not in t:
                        t["entry_price"] = t["level"]
                    if "stopLevel" in t and "stop_loss" not in t:
                        t["stop_loss"] = t.get("stopLevel") or 0
                    if "limitLevel" in t and "take_profit" not in t:
                        t["take_profit"] = t.get("limitLevel") or 0
                    if "openedAt" in t and "opened_at" not in t:
                        t["opened_at"] = t["openedAt"]
            elif isinstance(broker_df, list):
                broker_trades = broker_df
            else:
                broker_trades = []

            # Safety: if broker returned 0 trades but DB has positions,
            # the broker connection likely failed — abort reconciliation
            # to prevent falsely closing all DB positions.
            if len(broker_trades) == 0 and len(db_positions) > 0:
                connected = getattr(self.broker, '_connected', None)
                if connected is False or connected is None:
                    logger.warning(
                        "[Reconciliation] Broker not connected — skipping "
                        f"(DB has {len(db_positions)} positions, would wrongly close all)"
                    )
                    report["errors"].append("Broker not connected, skipped reconciliation")
                    return report

            report["broker_positions"] = len(broker_trades)
            broker_by_deal = {t.get("deal_id", ""): t for t in broker_trades}

            # 3. Match and reconcile
            all_deal_ids = set(db_by_deal.keys()) | set(broker_by_deal.keys())

            for deal_id in all_deal_ids:
                if not deal_id:
                    continue

                in_db = deal_id in db_by_deal
                in_broker = deal_id in broker_by_deal

                if in_db and in_broker:
                    report["matched"] += 1

                elif in_db and not in_broker:
                    # Position closed at broker but DB still shows open
                    logger.warning(
                        f"[Reconciliation] deal={deal_id} in DB but not OANDA — marking closed"
                    )
                    try:
                        self.position_store.close_position(
                            deal_id=deal_id,
                            exit_price=0,
                            pnl=0,
                        )
                        report["db_only_closed"] += 1
                    except Exception as e:
                        report["errors"].append(
                            f"Failed to close DB position {deal_id}: {e}"
                        )

                elif not in_db and in_broker:
                    # Position exists at broker but not in DB — add it
                    broker_trade = broker_by_deal[deal_id]
                    logger.warning(
                        f"[Reconciliation] deal={deal_id} in OANDA but not DB — adding"
                    )
                    try:
                        trade_info = {
                            "deal_id": deal_id,
                            "instrument": broker_trade.get("instrument", ""),
                            "direction": broker_trade.get("direction", ""),
                            "entry_price": broker_trade.get("entry_price", 0),
                            "size": broker_trade.get("size", 0),
                            "stop_loss": broker_trade.get("stop_loss", 0),
                            "take_profit": broker_trade.get("take_profit", 0),
                            "opened_at": broker_trade.get("opened_at", datetime.now().isoformat()),
                        }
                        self.position_store.save_position(trade_info)
                        report["broker_only_added"] += 1
                    except Exception as e:
                        report["errors"].append(
                            f"Failed to add broker position {deal_id}: {e}"
                        )

        except Exception as e:
            logger.error(f"[Reconciliation] Fatal error: {e}")
            report["errors"].append(f"Fatal: {e}")

        # Summary log
        logger.info(
            f"[Reconciliation] Complete: "
            f"matched={report['matched']}, "
            f"db_closed={report['db_only_closed']}, "
            f"broker_added={report['broker_only_added']}, "
            f"errors={len(report['errors'])}"
        )

        return report

    def reconcile_periodic(self) -> dict[str, Any]:
        """Lightweight periodic consistency check.

        Returns:
            Reconciliation report.
        """
        return self.reconcile_on_startup()
