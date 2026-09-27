"""SQLite persistence of the day-ahead prediction next to the actual price (#36).

Mixed into ``LearningStorage`` so the learning database keeps a single
connection and write lock. Self-learning removes a slot's stored predictions
once it has scored them; the ``evaluation`` table keeps one of them per slot
(the one made closest to ``EVALUATION_LEAD_HOURS`` ahead) next to the actual
price, for a few days, so Home Assistant can chart predicted against actual.
"""

import sqlite3
from typing import Any

from .storage_base import StorageMixinBase


class EvaluationStorageMixin(StorageMixinBase):
    """Evaluation table operations for ``LearningStorage``."""

    @staticmethod
    def _create_evaluation_schema(conn: sqlite3.Connection) -> None:
        """Create the evaluation table (idempotent)."""
        conn.execute(
            """CREATE TABLE IF NOT EXISTS evaluation (
                   timestamp   TEXT    PRIMARY KEY,
                   predicted   REAL    NOT NULL,
                   actual      REAL    NOT NULL,
                   lead_hours  REAL    NOT NULL
               )"""
        )

    def upsert_evaluation(
        self, timestamp: str, predicted: float, actual: float, lead_hours: float
    ) -> None:
        """Store a slot's evaluated prediction, replacing an earlier one (blocking).

        Args:
            timestamp: The slot's UTC key (``utc_slot_key``).
            predicted: The prediction (raw spot price, currency/kWh).
            actual: The actual price, in the same unit.
            lead_hours: How long before the slot the prediction was made.
        """
        with self._lock:
            conn = self._ensure_conn()
            conn.execute(
                """INSERT INTO evaluation (timestamp, predicted, actual, lead_hours)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(timestamp) DO UPDATE SET
                       predicted = excluded.predicted,
                       actual = excluded.actual,
                       lead_hours = excluded.lead_hours""",
                (timestamp, predicted, actual, lead_hours),
            )
            conn.commit()

    def get_evaluation(self, since: str) -> list[dict[str, Any]]:
        """Return the evaluated slots from ``since`` (a UTC key) on, in order (blocking)."""
        with self._lock:
            conn = self._ensure_conn()
            rows = conn.execute(
                """SELECT timestamp, predicted, actual, lead_hours
                   FROM evaluation
                   WHERE julianday(timestamp) >= julianday(?)
                   ORDER BY julianday(timestamp)""",
                (since,),
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
