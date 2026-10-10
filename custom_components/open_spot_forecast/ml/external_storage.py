"""SQLite persistence of external price forecasts and their accuracy (#120).

Mixed into ``LearningStorage`` so the learning database keeps a single
connection and write lock. ``external_forecasts`` holds the forecasts other
integrations publish (Stromligning's forecast sensor, Energi Data Service's
forecast attribute), read at every forecast run next to the model's own
predictions: one row per (source, slot, stored_at), as the raw spot price in
currency/kWh. Self-learning scores a slot's rows when its actual price is
known and removes them, like the model's predictions; their errors are kept
as daily sums per source and lead-time bucket in ``external_accuracy``, the
shape of ``lead_time_accuracy``.

The daily sums cannot score a blend of the model with a source: that needs
both errors of the same slot. ``external_slot_errors`` keeps them (#157), one
row per (slot, lead-time bucket, source) with the mean signed error of the
forecasts in the bucket, the model's own under ``EXTERNAL_MODEL_SOURCE``.
"""

from collections.abc import Sequence
from typing import Any

from .storage_base import StorageMixinBase

EXTERNAL_SCHEMA_SQL = """
    CREATE TABLE IF NOT EXISTS external_forecasts (
        source     TEXT NOT NULL,
        start      TEXT NOT NULL,
        stored_at  TEXT NOT NULL,
        price      REAL NOT NULL,
        PRIMARY KEY (start, source, stored_at)
    );
    CREATE TABLE IF NOT EXISTS external_accuracy (
        source          TEXT    NOT NULL,
        date            TEXT    NOT NULL,
        bucket          TEXT    NOT NULL,
        samples         INTEGER NOT NULL,
        sum_error       REAL    NOT NULL,
        sum_abs_error   REAL    NOT NULL,
        sum_sq_error    REAL    NOT NULL,
        PRIMARY KEY (source, date, bucket)
    );
    CREATE TABLE IF NOT EXISTS external_slot_errors (
        start    TEXT    NOT NULL,
        bucket   TEXT    NOT NULL,
        source   TEXT    NOT NULL,
        samples  INTEGER NOT NULL,
        error    REAL    NOT NULL,
        PRIMARY KEY (start, bucket, source)
    ) WITHOUT ROWID;
"""


class ExternalForecastStorageMixin(StorageMixinBase):
    """External forecast and accuracy table operations for ``LearningStorage``."""

    def insert_external_forecasts(
        self, source: str, stored_at: str, rows: Sequence[tuple[str, float]]
    ) -> None:
        """Store one reading of a source's forecast (blocking).

        Args:
            source: The source's name (the entity it is read from).
            stored_at: When the forecast was read (ISO, with its UTC offset).
            rows: ``(slot's UTC key, raw spot price in currency/kWh)``.
        """
        with self._lock:
            conn = self._ensure_conn()
            conn.executemany(
                """INSERT OR REPLACE INTO external_forecasts
                   (source, start, stored_at, price) VALUES (?, ?, ?, ?)""",
                [(source, start, stored_at, price) for start, price in rows],
            )
            conn.commit()

    def find_external_forecasts(self, start: str) -> list[dict[str, Any]]:
        """Return every stored forecast for a slot, oldest first (blocking).

        Args:
            start: The slot's UTC key (``utc_slot_key``).

        Returns:
            ``{"source", "start", "stored_at", "price"}`` rows.
        """
        with self._lock:
            conn = self._ensure_conn()
            rows = conn.execute(
                """SELECT source, start, stored_at, price
                   FROM external_forecasts
                   WHERE start = ?
                   ORDER BY source, stored_at""",
                (start,),
            ).fetchall()
        return [
            {"source": r[0], "start": r[1], "stored_at": r[2], "price": r[3]}
            for r in rows
        ]

    def delete_external_forecasts(self, start: str) -> int:
        """Delete a slot's stored forecasts once they are scored (blocking).

        Returns:
            Number of rows deleted.
        """
        with self._lock:
            conn = self._ensure_conn()
            cursor = conn.execute(
                "DELETE FROM external_forecasts WHERE start = ?", (start,)
            )
            conn.commit()
            return cursor.rowcount

    def delete_external_forecasts_stored_before(self, cutoff: str) -> int:
        """Delete the forecasts read before ``cutoff`` (ISO or UTC key) (blocking).

        Returns:
            Number of rows deleted.
        """
        with self._lock:
            conn = self._ensure_conn()
            cursor = conn.execute(
                "DELETE FROM external_forecasts"
                " WHERE julianday(stored_at) < julianday(?)",
                (cutoff,),
            )
            conn.commit()
            return cursor.rowcount

    def count_external_forecasts(self) -> int:
        """Count the stored external forecast rows (blocking)."""
        with self._lock:
            conn = self._ensure_conn()
            row = conn.execute("SELECT COUNT(*) FROM external_forecasts").fetchone()
        return row[0] if row else 0

    def add_external_errors(
        self, date_str: str, source: str, errors_by_bucket: dict[str, list[float]]
    ) -> None:
        """Add a source's signed errors to the running sums of a slot date (blocking).

        Args:
            date_str: Local date of the scored slot (``YYYY-MM-DD``).
            source: The source's name.
            errors_by_bucket: Signed errors (forecast - actual) per bucket key.
        """
        rows = [
            (
                source,
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
                """INSERT INTO external_accuracy
                   (source, date, bucket, samples, sum_error, sum_abs_error,
                    sum_sq_error)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(source, date, bucket) DO UPDATE SET
                       samples = samples + excluded.samples,
                       sum_error = sum_error + excluded.sum_error,
                       sum_abs_error = sum_abs_error + excluded.sum_abs_error,
                       sum_sq_error = sum_sq_error + excluded.sum_sq_error""",
                rows,
            )
            conn.commit()

    def get_external_error_sums(
        self, since_date: str
    ) -> dict[str, dict[str, tuple[int, float, float, float]]]:
        """Return error sums per source and bucket from ``since_date`` on (blocking).

        Args:
            since_date: First slot date (``YYYY-MM-DD``) inside the window.

        Returns:
            ``{source: {bucket: (samples, sum_error, sum_abs_error,
            sum_sq_error)}}``.
        """
        with self._lock:
            conn = self._ensure_conn()
            rows = conn.execute(
                """SELECT source, bucket, SUM(samples), SUM(sum_error),
                          SUM(sum_abs_error), SUM(sum_sq_error)
                   FROM external_accuracy
                   WHERE date >= ?
                   GROUP BY source, bucket""",
                (since_date,),
            ).fetchall()
        sums: dict[str, dict[str, tuple[int, float, float, float]]] = {}
        for source, bucket, samples, error, abs_error, sq_error in rows:
            sums.setdefault(source, {})[bucket] = (
                int(samples),
                float(error),
                float(abs_error),
                float(sq_error),
            )
        return sums

    def delete_external_accuracy_before(self, cutoff_date: str) -> int:
        """Delete the accuracy rows of slot dates before ``cutoff_date`` (blocking).

        Returns:
            Number of rows deleted.
        """
        with self._lock:
            conn = self._ensure_conn()
            cursor = conn.execute(
                "DELETE FROM external_accuracy WHERE date < ?", (cutoff_date,)
            )
            conn.commit()
            return cursor.rowcount

    def upsert_external_slot_errors(
        self, start: str, rows: Sequence[tuple[str, str, int, float]]
    ) -> None:
        """Store a scored slot's mean errors per bucket and source (blocking).

        Args:
            start: The slot's UTC key (``utc_slot_key``).
            rows: ``(bucket, source, number of forecasts, mean signed error)``,
                the error (forecast - actual) in currency/kWh.
        """
        if not rows:
            return
        with self._lock:
            conn = self._ensure_conn()
            conn.executemany(
                """INSERT OR REPLACE INTO external_slot_errors
                   (start, bucket, source, samples, error) VALUES (?, ?, ?, ?, ?)""",
                [(start, *row) for row in rows],
            )
            conn.commit()

    def get_external_slot_errors(self, since: str) -> list[dict[str, Any]]:
        """Return the per-slot errors from ``since`` (a UTC key) on (blocking).

        Returns:
            ``{"start", "bucket", "source", "samples", "error"}`` rows, by slot.
        """
        with self._lock:
            conn = self._ensure_conn()
            rows = conn.execute(
                """SELECT start, bucket, source, samples, error
                   FROM external_slot_errors
                   WHERE start >= ?
                   ORDER BY start, bucket, source""",
                (since,),
            ).fetchall()
        return [
            {
                "start": r[0],
                "bucket": r[1],
                "source": r[2],
                "samples": r[3],
                "error": r[4],
            }
            for r in rows
        ]

    def delete_external_slot_errors_before(self, cutoff: str) -> int:
        """Delete the per-slot errors of slots before ``cutoff`` (a UTC key) (blocking).

        UTC keys have a fixed width, so they are compared as text and the
        primary key serves the range.

        Returns:
            Number of rows deleted.
        """
        with self._lock:
            conn = self._ensure_conn()
            cursor = conn.execute(
                "DELETE FROM external_slot_errors WHERE start < ?", (cutoff,)
            )
            conn.commit()
            return cursor.rowcount
