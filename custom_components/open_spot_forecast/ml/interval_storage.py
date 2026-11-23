"""SQLite persistence of the errors behind the prediction intervals (#280).

Mixed into ``LearningStorage`` so the learning database keeps a single
connection and write lock. ``interval_errors`` holds one row per scored
prediction: the slot's local date, its lead-time bucket, its slot of the day
by wall clock (0-95) and the raw model's signed error (``price +
bias_offset - actual``). The rows of the last ``INTERVAL_WINDOW_DAYS`` days
are the band's sample (``ml/intervals.py``); older ones are pruned before
every forecast run.
"""

from collections.abc import Mapping, Sequence

from .storage_base import StorageMixinBase

INTERVAL_SCHEMA_SQL = """
    CREATE TABLE IF NOT EXISTS interval_errors (
        date    TEXT    NOT NULL,
        bucket  TEXT    NOT NULL,
        slot    INTEGER NOT NULL,
        error   REAL    NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_interval_errors_date
        ON interval_errors(date);
"""


class IntervalStorageMixin(StorageMixinBase):
    """Prediction interval error table operations for ``LearningStorage``."""

    def add_interval_errors(
        self, date_str: str, slot: int, errors_by_bucket: Mapping[str, Sequence[float]]
    ) -> None:
        """Store a scored slot's raw errors per lead-time bucket (blocking).

        Args:
            date_str: Local date of the slot (``YYYY-MM-DD``).
            slot: The slot of the local day by wall clock (0-95).
            errors_by_bucket: Raw errors (predicted + offset - actual) per bucket.
        """
        rows = [
            (date_str, bucket, slot, float(error))
            for bucket, errors in errors_by_bucket.items()
            for error in errors
        ]
        if not rows:
            return
        with self._lock:
            conn = self._ensure_conn()
            conn.executemany(
                """INSERT INTO interval_errors (date, bucket, slot, error)
                   VALUES (?, ?, ?, ?)""",
                rows,
            )
            conn.commit()

    def get_interval_errors(self, since_date: str) -> list[tuple[str, str, int, float]]:
        """Return ``(date, bucket, slot, error)`` from since_date on (blocking).

        Args:
            since_date: First slot date (``YYYY-MM-DD``) inside the window.
        """
        with self._lock:
            conn = self._ensure_conn()
            rows = conn.execute(
                """SELECT date, bucket, slot, error FROM interval_errors
                   WHERE date >= ?""",
                (since_date,),
            ).fetchall()
        return [(str(row[0]), str(row[1]), int(row[2]), float(row[3])) for row in rows]

    def delete_interval_errors_before(self, cutoff_date: str) -> int:
        """Delete the errors of slot dates before cutoff_date (blocking).

        Returns:
            Number of rows deleted.
        """
        with self._lock:
            conn = self._ensure_conn()
            cursor = conn.execute(
                "DELETE FROM interval_errors WHERE date < ?", (cutoff_date,)
            )
            conn.commit()
            return cursor.rowcount
