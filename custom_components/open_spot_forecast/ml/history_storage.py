"""Bulk reads of stored weather snapshots and Nordpool prognoses.

Mixed into ``LearningStorage`` so the learning database keeps a single
connection and write lock. Training reads both tables once per fit and
matches rows to slots in Python (``ml/training_inputs.py``), instead of one
query per training row.
"""

import sqlite3
import threading
from typing import Any


class HistoryStorageMixin:
    """Training-history reads for ``LearningStorage``."""

    # Provided by LearningStorage.
    _lock: threading.Lock

    def _ensure_conn(self) -> sqlite3.Connection:
        raise NotImplementedError

    def load_weather_history(self) -> list[dict[str, Any]]:
        """Return every stored weather snapshot, oldest first (blocking).

        Each row has ``timestamp`` (ISO; older rows are naive local time),
        ``temperature``, ``wind_speed`` (m/s), ``wind_direction``,
        ``cloud_coverage`` and ``humidity``; any value may be None.
        """
        with self._lock:
            rows = (
                self._ensure_conn()
                .execute(
                    """SELECT timestamp, temperature, wind_speed, wind_direction,
                              cloud_coverage, humidity
                       FROM weather_history ORDER BY timestamp"""
                )
                .fetchall()
            )
        keys = (
            "timestamp",
            "temperature",
            "wind_speed",
            "wind_direction",
            "cloud_coverage",
            "humidity",
        )
        return [dict(zip(keys, row, strict=True)) for row in rows]

    def load_nordpool_history(self) -> list[dict[str, Any]]:
        """Return every stored Nordpool prognosis row, oldest first (blocking).

        Each row has ``timestamp`` (UTC ISO, one row per hour), ``consumption``,
        ``solar``, ``wind_offshore`` and ``wind_onshore`` in MW; any value may
        be None.
        """
        with self._lock:
            rows = (
                self._ensure_conn()
                .execute(
                    """SELECT timestamp, consumption, solar, wind_offshore,
                              wind_onshore
                       FROM nordpool_prognoses ORDER BY timestamp"""
                )
                .fetchall()
            )
        keys = ("timestamp", "consumption", "solar", "wind_offshore", "wind_onshore")
        return [dict(zip(keys, row, strict=True)) for row in rows]
