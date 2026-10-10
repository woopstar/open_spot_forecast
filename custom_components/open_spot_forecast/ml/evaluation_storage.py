"""SQLite persistence of the day-ahead prediction next to the actual price (#36).

Mixed into ``LearningStorage`` so the learning database keeps a single
connection and write lock. Self-learning removes a slot's stored predictions
once it has scored them; the ``evaluation`` table keeps a few of them per
slot, one per lead time in ``EVALUATION_LEAD_TIMES`` (the one made closest to
that long ahead, #113), next to the actual price, for a few days, so Home
Assistant can chart predicted against actual.

Before #113 the table held one row per slot, the ``EVALUATION_LEAD_HOURS``
one. Such a table is rebuilt once at startup with ``target_hours`` in its
key; its rows are kept as that lead time's.
"""

import logging
import sqlite3
from typing import Any

from ..const import EVALUATION_LEAD_HOURS
from .storage_base import StorageMixinBase

_LOGGER = logging.getLogger(__name__)

EVALUATION_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS evaluation (
        timestamp     TEXT    NOT NULL,
        target_hours  REAL    NOT NULL,
        predicted     REAL    NOT NULL,
        actual        REAL    NOT NULL,
        lead_hours    REAL    NOT NULL,
        PRIMARY KEY (timestamp, target_hours)
    )
"""

# REAL columns are compared with a tolerance, never for equality
_AT_TARGET = "ABS(target_hours - ?) < 1e-6"


def migrate_evaluation_to_lead_times(conn: sqlite3.Connection) -> None:
    """Key the evaluation rows by (slot, lead time) once (#113).

    A table from before #113 has no ``target_hours`` column: its rows are
    kept as the ``EVALUATION_LEAD_HOURS`` snapshots. Detected by the table's
    columns, so it runs once and needs no schema version.

    Args:
        conn: Open learning database connection; the caller commits.
    """
    columns = [c[1] for c in conn.execute("PRAGMA table_info(evaluation)")]
    if not columns or "target_hours" in columns:
        return
    conn.execute("ALTER TABLE evaluation RENAME TO evaluation_single")
    conn.execute(EVALUATION_TABLE_SQL)
    kept = conn.execute(
        "INSERT INTO evaluation (timestamp, target_hours, predicted, actual, lead_hours) "
        "SELECT timestamp, ?, predicted, actual, lead_hours FROM evaluation_single",
        (EVALUATION_LEAD_HOURS,),
    ).rowcount
    conn.execute("DROP TABLE evaluation_single")
    if kept:
        _LOGGER.info(
            "The evaluation now keeps a prediction per lead time: kept %d slots "
            "as the %g h ones",
            kept,
            EVALUATION_LEAD_HOURS,
        )


class EvaluationStorageMixin(StorageMixinBase):
    """Evaluation table operations for ``LearningStorage``."""

    @staticmethod
    def _create_evaluation_schema(conn: sqlite3.Connection) -> None:
        """Create the evaluation table, or rebuild one from before #113 (idempotent)."""
        migrate_evaluation_to_lead_times(conn)
        conn.execute(EVALUATION_TABLE_SQL)

    def upsert_evaluation(
        self,
        timestamp: str,
        predicted: float,
        actual: float,
        lead_hours: float,
        target_hours: float = EVALUATION_LEAD_HOURS,
    ) -> None:
        """Store a slot's evaluated prediction, replacing an earlier one (blocking).

        Args:
            timestamp: The slot's UTC key (``utc_slot_key``).
            predicted: The prediction (raw spot price, currency/kWh).
            actual: The actual price, in the same unit.
            lead_hours: How long before the slot the prediction was made.
            target_hours: The lead time the prediction is the snapshot of
                (one of ``EVALUATION_LEAD_TIMES``).
        """
        with self._lock:
            conn = self._ensure_conn()
            conn.execute(
                """INSERT INTO evaluation
                       (timestamp, target_hours, predicted, actual, lead_hours)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(timestamp, target_hours) DO UPDATE SET
                       predicted = excluded.predicted,
                       actual = excluded.actual,
                       lead_hours = excluded.lead_hours""",
                (timestamp, target_hours, predicted, actual, lead_hours),
            )
            conn.commit()

    def get_evaluation(
        self, since: str, target_hours: float = EVALUATION_LEAD_HOURS
    ) -> list[dict[str, Any]]:
        """Return a lead time's evaluated slots from ``since`` on, in order (blocking).

        Args:
            since: The first slot's UTC key.
            target_hours: The lead time whose snapshots are returned.
        """
        with self._lock:
            conn = self._ensure_conn()
            rows = conn.execute(
                f"""SELECT timestamp, predicted, actual, lead_hours
                    FROM evaluation
                    WHERE julianday(timestamp) >= julianday(?) AND {_AT_TARGET}
                    ORDER BY julianday(timestamp)""",
                (since, target_hours),
            ).fetchall()
        return [
            {
                "timestamp": row[0],
                "predicted": row[1],
                "actual": row[2],
                "lead_hours": row[3],
            }
            for row in rows
        ]

    def delete_evaluation_before(self, cutoff: str) -> int:
        """Delete the slots before ``cutoff`` (a UTC key) (blocking).

        Returns:
            Number of rows deleted.
        """
        with self._lock:
            conn = self._ensure_conn()
            cursor = conn.execute(
                "DELETE FROM evaluation WHERE julianday(timestamp) < julianday(?)",
                (cutoff,),
            )
            conn.commit()
            return cursor.rowcount
