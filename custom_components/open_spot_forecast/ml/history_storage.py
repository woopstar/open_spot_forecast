"""SQLite persistence for the model's training history.

Mixed into ``LearningStorage`` so the learning database keeps a single
connection and write lock. Covers the ``weather_history`` snapshots, the
``nordpool_prognoses`` rows and the daily ``price_history``. Training reads
the weather and Nordpool tables once per fit and matches rows to slots in
Python (``ml/training_inputs.py``), instead of one query per training row.

Weather snapshots are keyed by their UTC slot start (``utc_slot_key``);
Nordpool rows by the UTC hour Nordpool publishes. The single-row lookups and
the pruning compare timestamps as SQLite julian days on both sides, so a row
and a query only have to be the same instant, not the same string. Window
bounds are computed in Python and compared with ``julianday()`` of the stored
key, so a row exactly on a bound is always inside (#46).
"""

import json
from datetime import datetime, timedelta
from typing import Any

from homeassistant.util import dt as dt_util

from ..time_slots import UTC_KEY_FORMAT, floor_to_slot, parse_utc, utc_slot_key
from .storage_base import StorageMixinBase

# How far from a looked-up time a weather snapshot may be (inclusive)
_WEATHER_WINDOW = timedelta(minutes=30)
# Nordpool publishes one prognosis row per hour
_NORDPOOL_ROW_MINUTES = 60


def _utc_key(moment: datetime) -> str:
    """Return a UTC datetime in the stored key format."""
    return moment.strftime(UTC_KEY_FORMAT)


def _utc_cutoff(max_age_days: int) -> str:
    """Return the UTC key ``max_age_days`` before now (the pruning cutoff)."""
    return utc_slot_key(dt_util.utcnow() - timedelta(days=max_age_days))


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
        """Store a weather snapshot for a given timestamp (blocking).

        Snapshots score the local weather forecast (confidence); they are not
        training data (#23), so storing one does not trigger a retrain.
        """
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

    def find_weather_for_timestamp(self, timestamp: str) -> dict[str, float] | None:
        """Find the snapshot closest to a timestamp, within 30 minutes (blocking).

        A slot's own snapshot is 0 minutes from its start, so it wins over
        its neighbours; a missing slot falls back to a snapshot at most 30
        minutes away, and nothing further.

        Args:
            timestamp: ISO timestamp, with any UTC offset (naive = Home
                Assistant local time).
        """
        moment = parse_utc(timestamp)
        if moment is None:
            return None
        with self._lock:
            conn = self._ensure_conn()
            row = conn.execute(
                """SELECT temperature, wind_speed, wind_direction,
                          cloud_coverage, humidity, solar_power
                   FROM weather_history
                   WHERE julianday(timestamp) BETWEEN julianday(?) AND julianday(?)
                   ORDER BY ABS(julianday(timestamp) - julianday(?))
                   LIMIT 1""",
                (
                    _utc_key(moment - _WEATHER_WINDOW),
                    _utc_key(moment + _WEATHER_WINDOW),
                    _utc_key(moment),
                ),
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
        cutoff = _utc_cutoff(max_age_days)
        with self._lock:
            conn = self._ensure_conn()
            cursor = conn.execute(
                "DELETE FROM weather_history WHERE julianday(timestamp) < julianday(?)",
                (cutoff,),
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

        Each row has ``timestamp`` (UTC slot start, ``…Z``, see
        ``utc_slot_key``),
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

    def find_nordpool_for_timestamp(self, timestamp: str) -> dict[str, float] | None:
        """Find the prognosis for a timestamp's UTC hour (blocking).

        The same row training uses for every slot of the hour
        (``TrainingInputs``): the first stored in the hour, never a
        neighbouring hour's.

        Args:
            timestamp: ISO timestamp, with any UTC offset (naive = Home
                Assistant local time).
        """
        moment = parse_utc(timestamp)
        if moment is None:
            return None
        hour = floor_to_slot(moment, _NORDPOOL_ROW_MINUTES)
        with self._lock:
            conn = self._ensure_conn()
            row = conn.execute(
                """SELECT consumption, solar, wind_offshore, wind_onshore
                   FROM nordpool_prognoses
                   WHERE julianday(timestamp) >= julianday(?)
                     AND julianday(timestamp) < julianday(?)
                   ORDER BY julianday(timestamp)
                   LIMIT 1""",
                (
                    _utc_key(hour),
                    _utc_key(hour + timedelta(minutes=_NORDPOOL_ROW_MINUTES)),
                ),
            ).fetchone()

        if row is None:
            return None
        return {
            "consumption": row[0],
            "solar": row[1],
            "wind_offshore": row[2],
            "wind_onshore": row[3],
        }

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

    def delete_old_prices(self, before: str) -> int:
        """Delete the price days before ``before`` (YYYY-MM-DD) (blocking)."""
        with self._lock:
            conn = self._ensure_conn()
            cursor = conn.execute("DELETE FROM price_history WHERE date < ?", (before,))
            conn.commit()
            return cursor.rowcount
