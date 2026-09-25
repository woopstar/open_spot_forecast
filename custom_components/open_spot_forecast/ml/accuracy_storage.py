"""SQLite persistence for live forecast accuracy per lead time.

Mixed into ``LearningStorage`` so the learning database keeps a single
connection and write lock. The ``lead_time_accuracy`` table holds one row per
(slot date, lead-time bucket) with running error sums, which keeps it tiny
(buckets x window days) while still letting MAE, RMSE and bias be computed
over any rolling window of whole days.
"""

import sqlite3

from .storage_base import StorageMixinBase


class LeadTimeAccuracyStorageMixin(StorageMixinBase):
    """Lead-time accuracy table operations for ``LearningStorage``."""

    @staticmethod
    def _create_lead_time_accuracy_schema(conn: sqlite3.Connection) -> None:
        """Create the lead_time_accuracy table (idempotent)."""
        conn.execute(
            """CREATE TABLE IF NOT EXISTS lead_time_accuracy (
                   date            TEXT    NOT NULL,
                   bucket          TEXT    NOT NULL,
                   samples         INTEGER NOT NULL,
                   sum_error       REAL    NOT NULL,
                   sum_abs_error   REAL    NOT NULL,
                   sum_sq_error    REAL    NOT NULL,
                   PRIMARY KEY (date, bucket)
               )"""
        )

    def add_lead_time_errors(
        self, date_str: str, errors_by_bucket: dict[str, list[float]]
    ) -> None:
        """Add signed prediction errors to the running sums for a slot date (blocking).

        Args:
            date_str: Local date of the matched slot (``YYYY-MM-DD``).
            errors_by_bucket: Signed errors (predicted - actual) per bucket key.
        """
        rows = [
            (
                date_str,
                bucket,
                len(errors),
                sum(errors),
                sum(abs(e) for e in errors),
                sum(e * e for e in errors),
            )
            for bucket, errors in errors_by_bucket.items()
            if errors
        ]
        if not rows:
            return
        with self._lock:
            conn = self._ensure_conn()
            conn.executemany(
                """INSERT INTO lead_time_accuracy
                   (date, bucket, samples, sum_error, sum_abs_error, sum_sq_error)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(date, bucket) DO UPDATE SET
                       samples = samples + excluded.samples,
                       sum_error = sum_error + excluded.sum_error,
                       sum_abs_error = sum_abs_error + excluded.sum_abs_error,
                       sum_sq_error = sum_sq_error + excluded.sum_sq_error""",
                rows,
            )
            conn.commit()

    def get_lead_time_error_sums(
        self, since_date: str
    ) -> dict[str, tuple[int, float, float, float]]:
        """Return error sums per bucket for slot dates on or after since_date (blocking).

        Args:
            since_date: First slot date (``YYYY-MM-DD``) inside the window.

        Returns:
            ``{bucket: (samples, sum_error, sum_abs_error, sum_sq_error)}``.
        """
        with self._lock:
            conn = self._ensure_conn()
            rows = conn.execute(
                """SELECT bucket, SUM(samples), SUM(sum_error),
                          SUM(sum_abs_error), SUM(sum_sq_error)
                   FROM lead_time_accuracy
                   WHERE date >= ?
                   GROUP BY bucket""",
                (since_date,),
            ).fetchall()
        return {r[0]: (int(r[1]), float(r[2]), float(r[3]), float(r[4])) for r in rows}

    def delete_lead_time_accuracy_before(self, cutoff_date: str) -> int:
        """Delete rows for slot dates before cutoff_date (blocking).

        Returns:
            Number of rows deleted.
        """
        with self._lock:
            conn = self._ensure_conn()
            cursor = conn.execute(
                "DELETE FROM lead_time_accuracy WHERE date < ?", (cutoff_date,)
            )
            conn.commit()
            return cursor.rowcount
