"""SQLite persistence for predictions awaiting self-learning comparison.

Mixed into ``LearningStorage`` so the learning database keeps a single
connection and write lock. Each forecast run inserts its predictions one row
at a time; self-learning (and the startup catch-up) matches them against the
actual price once their slot has passed and then removes them.
"""

from datetime import timedelta
from typing import Any

from homeassistant.util import dt as dt_util

from .storage_base import StorageMixinBase


class PredictionStorageMixin(StorageMixinBase):
    """Predictions table operations for ``LearningStorage``."""

    def insert_prediction(
        self,
        start: str,
        price: float,
        confidence: float,
        hour: int,
        minute: int,
        stored_at: str,
        forecast_temp: float | None = None,
        forecast_wind: float | None = None,
        forecast_cloud: float | None = None,
    ) -> int | None:
        """Insert a single prediction entry (blocking).

        Optionally stores the weather forecast values that were used,
        so they can later be compared against actual measurements.

        Returns the auto-incremented id of the new row.
        """
        with self._lock:
            conn = self._ensure_conn()
            cursor = conn.execute(
                """INSERT INTO predictions
                   (start, price, confidence, hour, minute, stored_at,
                    forecast_temp, forecast_wind, forecast_cloud)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    start,
                    price,
                    confidence,
                    hour,
                    minute,
                    stored_at,
                    forecast_temp,
                    forecast_wind,
                    forecast_cloud,
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def delete_old_predictions(self, max_age_days: int) -> int:
        """Delete predictions older than max_age_days (blocking).

        Returns:
            Number of rows deleted.
        """
        cutoff = (dt_util.now() - timedelta(days=max_age_days)).isoformat()
        with self._lock:
            conn = self._ensure_conn()
            cursor = conn.execute(
                "DELETE FROM predictions WHERE stored_at < ?", (cutoff,)
            )
            conn.commit()
            return cursor.rowcount

    def find_predictions_for_timestamp(
        self, date_str: str, hour: int, minute: int
    ) -> list[dict[str, Any]]:
        """Find all predictions matching a specific date, hour, and minute.

        Matches on the date portion of the 'start' field plus hour/minute.
        (blocking)
        """
        prefix = date_str  # "YYYY-MM-DD"
        with self._lock:
            conn = self._ensure_conn()
            rows = conn.execute(
                """SELECT id, start, price, confidence, hour, minute, stored_at,
                          forecast_temp, forecast_wind, forecast_cloud
                   FROM predictions
                   WHERE start LIKE ? AND hour = ? AND minute = ?
                   ORDER BY stored_at ASC""",
                (f"{prefix}%", hour, minute),
            ).fetchall()

        if not rows:
            return []

        return [
            {
                "id": r[0],
                "start": r[1],
                "price": r[2],
                "confidence": r[3],
                "hour": r[4],
                "minute": r[5],
                "stored_at": r[6],
                "forecast_temp": r[7],
                "forecast_wind": r[8],
                "forecast_cloud": r[9],
            }
            for r in rows
        ]

    def get_pending_prediction_dates(self) -> list[str]:
        """Return distinct dates that have pending predictions (blocking).

        Used by catch-up learning to know which dates need replay.
        """
        with self._lock:
            conn = self._ensure_conn()
            rows = conn.execute(
                "SELECT DISTINCT SUBSTR(start, 1, 10) FROM predictions ORDER BY 1"
            ).fetchall()
        return [r[0] for r in rows if r[0]]

    def remove_prediction(self, prediction_id: int) -> None:
        """Remove a single prediction by its id (blocking)."""
        with self._lock:
            conn = self._ensure_conn()
            conn.execute("DELETE FROM predictions WHERE id = ?", (prediction_id,))
            conn.commit()

    def count_predictions(self) -> int:
        """Count total predictions in the database (blocking)."""
        with self._lock:
            conn = self._ensure_conn()
            row = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()
        return row[0] if row else 0
