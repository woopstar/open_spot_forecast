"""SQLite persistence for the model's training history.

Mixed into ``LearningStorage`` so the learning database keeps a single
connection and write lock. Covers the ``weather_history`` snapshots, the
``nordpool_prognoses`` rows and the daily ``price_history``. Training reads
the weather and Nordpool tables once per fit and matches rows to slots in
Python (``ml/training_inputs.py``), instead of one query per training row.
"""

import json
from datetime import timedelta
from typing import Any

from homeassistant.util import dt as dt_util

from .storage_base import StorageMixinBase


class HistoryStorageMixin(StorageMixinBase):
    """Weather, Nordpool prognosis and price history for ``LearningStorage``."""

    # ------------------------------------------------------------------
    # Weather history operations
    # ------------------------------------------------------------------

    def insert_weather_snapshot(
        self,
        timestamp: str,
        temperature: float | None,
        wind_speed: float | None,
        wind_direction: float | None,
        cloud_coverage: float | None,
        humidity: float | None,
        solar_power: float | None,
    ) -> None:
        """Store a weather snapshot for a given timestamp (blocking)."""
        with self._lock:
            conn = self._ensure_conn()
            conn.execute(
                """INSERT OR REPLACE INTO weather_history
                   (timestamp, temperature, wind_speed, wind_direction,
                    cloud_coverage, humidity, solar_power)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    timestamp,
                    temperature,
                    wind_speed,
                    wind_direction,
                    cloud_coverage,
                    humidity,
                    solar_power,
                ),
            )
            conn.commit()
            self.last_data_write = dt_util.utcnow()

    def find_weather_for_timestamp(self, timestamp: str) -> dict[str, float] | None:
        """Find weather snapshot closest to a given timestamp (blocking)."""
        with self._lock:
            conn = self._ensure_conn()
            row = conn.execute(
                """SELECT temperature, wind_speed, wind_direction,
                          cloud_coverage, humidity, solar_power
                   FROM weather_history
                   WHERE timestamp >= datetime(?, '-30 minutes')
                     AND timestamp <= datetime(?, '+30 minutes')
                   ORDER BY ABS(strftime('%s', timestamp) - strftime('%s', ?))
                   LIMIT 1""",
                (timestamp, timestamp, timestamp),
            ).fetchone()

        if row is None:
            return None
        return {
            "temperature": row[0],
            "wind_speed": row[1],
            "wind_direction": row[2],
            "cloud_coverage": row[3],
            "humidity": row[4],
            "solar_power": row[5],
        }

    def delete_old_weather(self, max_age_days: int = 30) -> int:
        """Delete weather snapshots older than max_age_days (blocking)."""
        cutoff = (dt_util.now() - timedelta(days=max_age_days)).isoformat()
        with self._lock:
            conn = self._ensure_conn()
            cursor = conn.execute(
                "DELETE FROM weather_history WHERE timestamp < ?", (cutoff,)
            )
            conn.commit()
            return cursor.rowcount

    def count_weather_snapshots(self) -> int:
        """Count total weather snapshots (blocking)."""
        with self._lock:
            conn = self._ensure_conn()
            row = conn.execute("SELECT COUNT(*) FROM weather_history").fetchone()
        return row[0] if row else 0

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

    # ------------------------------------------------------------------
    # Nordpool prognoses operations
    # ------------------------------------------------------------------

    def insert_nordpool_prognosis(
        self,
        timestamp: str,
        consumption: float | None,
        solar: float | None,
        wind_offshore: float | None,
        wind_onshore: float | None,
    ) -> None:
        """Store a Nordpool prognosis snapshot (blocking)."""
        with self._lock:
            conn = self._ensure_conn()
            conn.execute(
                """INSERT OR REPLACE INTO nordpool_prognoses
                   (timestamp, consumption, solar, wind_offshore, wind_onshore)
                   VALUES (?, ?, ?, ?, ?)""",
                (timestamp, consumption, solar, wind_offshore, wind_onshore),
            )
            conn.commit()

    def insert_nordpool_prognoses_batch(self, entries: list[dict]) -> None:
        """Store multiple Nordpool prognosis entries (blocking).

        Every forecast run re-sends the same prognoses, so rows are only
        rewritten when a value differs; last_data_write moves only on a
        real change.
        """
        with self._lock:
            conn = self._ensure_conn()
            cursor = conn.executemany(
                """INSERT INTO nordpool_prognoses
                   (timestamp, consumption, solar, wind_offshore, wind_onshore)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(timestamp) DO UPDATE SET
                       consumption = excluded.consumption,
                       solar = excluded.solar,
                       wind_offshore = excluded.wind_offshore,
                       wind_onshore = excluded.wind_onshore
                   WHERE consumption IS NOT excluded.consumption
                      OR solar IS NOT excluded.solar
                      OR wind_offshore IS NOT excluded.wind_offshore
                      OR wind_onshore IS NOT excluded.wind_onshore""",
                [
                    (
                        e.get("timestamp", ""),
                        e.get("consumption"),
                        e.get("solar"),
                        e.get("wind_offshore"),
                        e.get("wind_onshore"),
                    )
                    for e in entries
                ],
            )
            conn.commit()
            if cursor.rowcount > 0:
                self.last_data_write = dt_util.utcnow()

    def find_nordpool_for_timestamp(self, timestamp: str) -> dict[str, float] | None:
        """Find Nordpool prognosis closest to a timestamp (blocking)."""
        with self._lock:
            conn = self._ensure_conn()
            row = conn.execute(
                """SELECT consumption, solar, wind_offshore, wind_onshore
                   FROM nordpool_prognoses
                   WHERE timestamp >= datetime(?, '-1 hour')
                     AND timestamp <= datetime(?, '+1 hour')
                   ORDER BY ABS(strftime('%s', timestamp) - strftime('%s', ?))
                   LIMIT 1""",
                (timestamp, timestamp, timestamp),
            ).fetchone()

        if row is None:
            return None
        return {
            "consumption": row[0],
            "solar": row[1],
            "wind_offshore": row[2],
            "wind_onshore": row[3],
        }

    def delete_old_nordpool(self, max_age_days: int = 30) -> int:
        """Delete old Nordpool prognoses (blocking)."""
        cutoff = (dt_util.now() - timedelta(days=max_age_days)).isoformat()
        with self._lock:
            conn = self._ensure_conn()
            cursor = conn.execute(
                "DELETE FROM nordpool_prognoses WHERE timestamp < ?", (cutoff,)
            )
            conn.commit()
            return cursor.rowcount

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

    # ------------------------------------------------------------------
    # Price history operations
    # ------------------------------------------------------------------

    def save_price_history(self, price_history: list[dict]) -> None:
        """Persist price history to the database (blocking)."""
        with self._lock:
            conn = self._ensure_conn()
            for entry in price_history:
                date = entry.get("date")
                prices = entry.get("prices", [])
                if date is None:
                    continue
                conn.execute(
                    "INSERT OR REPLACE INTO price_history (date, prices) VALUES (?, ?)",
                    (date, json.dumps(prices)),
                )
            conn.commit()

    def load_price_history(self) -> list[dict]:
        """Load all price history from the database (blocking)."""
        with self._lock:
            conn = self._ensure_conn()
            rows = conn.execute(
                "SELECT date, prices FROM price_history ORDER BY date ASC"
            ).fetchall()
        return [
            {"date": date, "prices": json.loads(prices_json)}
            for date, prices_json in rows
        ]
