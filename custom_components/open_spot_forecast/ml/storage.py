"""Storage module for persisting ML learning data using SQLite."""

import contextlib
import json
import logging
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

_LOGGER = logging.getLogger(__name__)


class LearningStorage:
    """Handles persistence of learning data to a SQLite database.

    Uses a single persistent connection with WAL journal mode.
    All write operations are serialized via a threading lock.
    The database file is created automatically on first use.

    Schema:
      predictions     — per-interval forecasts awaiting self-learning comparison
      error_metrics   — per-hour error tracking (JSON-serialized arrays)
      bias_correction — per-hour multiplicative correction factors
      price_history   — historical daily prices for model training
      meta            — key/value pairs (training_samples, is_trained, hpo_counter)
    """

    def __init__(self, hass: HomeAssistant, region: str):
        """Initialize storage — opens persistent SQLite connection.

        If a legacy JSON learning file exists and the SQLite database
        is empty, data is auto-migrated on first startup.

        Args:
            hass: Home Assistant instance
            region: Price region (e.g., "DK1")
        """
        self.hass = hass
        self.region = region
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        # When training inputs (weather/Nordpool rows) last changed (UTC);
        # the predictor retrains when this is newer than its last training
        self.last_data_write: datetime | None = None

        storage_dir = Path(hass.config.path(".storage"))
        storage_dir.mkdir(exist_ok=True)

        self.db_path = storage_dir / f"open_spot_forecast_{region}_learning.db"
        self._json_path = storage_dir / f"open_spot_forecast_{region}_learning.json"

        # Ensure schema exists on first access
        self._ensure_conn()

        # Auto-migrate from legacy JSON file if needed
        if self._maybe_migrate_from_json():
            _LOGGER.info("Auto-migrated learning data from JSON to SQLite")

        _LOGGER.debug("Learning storage initialized at: %s", self.db_path)

    def _maybe_migrate_from_json(self) -> bool:
        """Auto-migrate data from legacy JSON file if the DB is empty.

        Called once during __init__. If the predictions table is empty
        and the old JSON learning file exists, imports all data and
        deletes the JSON file.

        Returns:
            True if migration was performed, False otherwise.
        """
        with self._lock:
            conn = self._ensure_conn()

            # Check if predictions table already has data
            count = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
            if count > 0:
                return False

        # No data in DB — check for legacy JSON file
        if not self._json_path.exists():
            return False

        try:
            with open(self._json_path, encoding="utf-8") as f:
                data = json.load(f)

            predictions = data.get("prediction_history", [])
            error_metrics = data.get("error_metrics", {})
            bias_correction = data.get("bias_correction", {})
            price_history = data.get("price_history", [])
            training_samples = data.get("training_samples", 0)
            is_trained = data.get("is_trained", False)

            _LOGGER.info(
                "Migrating from JSON: %d predictions, %d error metrics, "
                "%d bias corrections, %d price days",
                len(predictions),
                len(error_metrics),
                len(bias_correction),
                len(price_history),
            )

            # Import via save_all (atomic transaction)
            self.save_all(
                {
                    "prediction_history": predictions,
                    "error_metrics": error_metrics,
                    "bias_correction": bias_correction,
                    "price_history": price_history,
                    "training_samples": training_samples,
                    "is_trained": is_trained,
                }
            )

            imported = self.count_predictions()
            _LOGGER.info(
                "Migration complete: %d unique predictions imported",
                imported,
            )

            # Rename old JSON file as backup instead of deleting
            backup_path = self._json_path.with_suffix(".json.bak")
            self._json_path.rename(backup_path)
            _LOGGER.debug("Renamed old JSON to: %s", backup_path)

            return True

        except Exception as err:
            _LOGGER.error("Failed to migrate from JSON: %s", err)
            return False

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _ensure_conn(self) -> sqlite3.Connection:
        """Return the persistent connection, creating it on first call.

        Must be called while holding self._lock.
        """
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=FULL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._create_schema(self._conn)
        return self._conn

    def _create_schema(self, conn: sqlite3.Connection) -> None:
        """Create tables and indexes (idempotent)."""
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                start           TEXT    NOT NULL,
                price           REAL    NOT NULL,
                confidence      REAL,
                hour            INTEGER,
                minute          INTEGER,
                stored_at       TEXT    NOT NULL,
                forecast_temp   REAL,
                forecast_wind   REAL,
                forecast_cloud  REAL
            );

            CREATE INDEX IF NOT EXISTS idx_predictions_start
                ON predictions(start);

            CREATE INDEX IF NOT EXISTS idx_predictions_stored_at
                ON predictions(stored_at);

            CREATE INDEX IF NOT EXISTS idx_predictions_hour_minute
                ON predictions(hour, minute);

            CREATE TABLE IF NOT EXISTS error_metrics (
                hour        INTEGER PRIMARY KEY,
                data        TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS bias_correction (
                hour        INTEGER PRIMARY KEY,
                correction  REAL    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS price_history (
                date        TEXT    PRIMARY KEY,
                prices      TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS meta (
                key         TEXT    PRIMARY KEY,
                value       TEXT
            );

            CREATE TABLE IF NOT EXISTS weather_history (
                timestamp       TEXT    PRIMARY KEY,
                temperature     REAL,
                wind_speed      REAL,
                wind_direction  REAL,
                cloud_coverage  REAL,
                humidity        REAL,
                solar_power     REAL
            );

            CREATE INDEX IF NOT EXISTS idx_weather_timestamp
                ON weather_history(timestamp);

            CREATE TABLE IF NOT EXISTS nordpool_prognoses (
                timestamp       TEXT    PRIMARY KEY,
                consumption     REAL,
                solar           REAL,
                wind_offshore   REAL,
                wind_onshore    REAL
            );

            CREATE INDEX IF NOT EXISTS idx_nordpool_timestamp
                ON nordpool_prognoses(timestamp);

            CREATE TABLE IF NOT EXISTS volatility (
                slot    INTEGER PRIMARY KEY,
                mae     REAL    NOT NULL
            );
            """
        )

        # Schema migration: old DBs had start as PRIMARY KEY without id column
        cols = [c[1] for c in conn.execute("PRAGMA table_info(predictions)").fetchall()]
        if "id" not in cols:
            _LOGGER.info("Migrating predictions table to v2 (adding auto-increment id)")
            conn.executescript(
                """
                ALTER TABLE predictions RENAME TO predictions_old;
                CREATE TABLE predictions (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    start       TEXT    NOT NULL,
                    price       REAL    NOT NULL,
                    confidence  REAL,
                    hour        INTEGER,
                    minute      INTEGER,
                    stored_at   TEXT    NOT NULL
                );
                INSERT INTO predictions
                    (start, price, confidence, hour, minute, stored_at)
                SELECT start, price, confidence, hour, minute, stored_at
                FROM predictions_old;
                DROP TABLE predictions_old;
                CREATE INDEX IF NOT EXISTS idx_predictions_start
                    ON predictions(start);
                CREATE INDEX IF NOT EXISTS idx_predictions_stored_at
                    ON predictions(stored_at);
                CREATE INDEX IF NOT EXISTS idx_predictions_hour_minute
                    ON predictions(hour, minute);
                """
            )
            _LOGGER.info("Schema migration to v2 complete")

        # Schema migration v3: switch error_metrics/bias_correction
        # from hour-based keys (0-23) to slot-based keys (0-95).
        # SAFETY: only clear if there's evidence of old hour-based data
        # AND no slot-based data yet. This prevents accidental data loss
        # on re-runs (e.g. if the schema_version meta key is missing).
        meta_row = conn.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'"
        ).fetchone()
        schema_version = int(meta_row[0]) if meta_row else 0
        if schema_version < 3:
            # Check if error_metrics contains only hour keys 0-23
            max_key_row = conn.execute(
                "SELECT MAX(CAST(hour AS INTEGER)) FROM error_metrics"
            ).fetchone()
            max_key = max_key_row[0] if max_key_row else None
            has_old_data = max_key is not None and max_key <= 23
            has_slot_data = max_key is not None and max_key > 23

            if has_old_data and not has_slot_data:
                _LOGGER.info(
                    "Schema v3 migration: clearing %d old hour-based metrics (max_key=%s)",
                    conn.execute("SELECT COUNT(*) FROM error_metrics").fetchone()[0],
                    max_key,
                )
                conn.execute("DELETE FROM error_metrics")
                conn.execute("DELETE FROM bias_correction")
            elif has_slot_data:
                _LOGGER.info(
                    "Schema v3 migration: slot-based data already present (max_key=%s), skipping clear",
                    max_key,
                )
            else:
                _LOGGER.info("Schema v3 migration: no data to migrate, skipping")

            conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', '3')"
            )

        # Schema migration v4: add forecast weather columns to predictions
        pred_cols = [
            c[1] for c in conn.execute("PRAGMA table_info(predictions)").fetchall()
        ]
        if "forecast_temp" not in pred_cols:
            _LOGGER.info("Migrating predictions table to v4 (adding forecast columns)")
            conn.executescript(
                """
                ALTER TABLE predictions ADD COLUMN forecast_temp REAL;
                ALTER TABLE predictions ADD COLUMN forecast_wind REAL;
                ALTER TABLE predictions ADD COLUMN forecast_cloud REAL;
                """
            )
            conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', '4')"
            )
            _LOGGER.info("Schema migration to v4 complete")

        conn.commit()

    def __del__(self) -> None:
        """Safety net: close connection if not already closed."""
        try:
            conn = getattr(self, "_conn", None)
            if conn is not None:
                conn.close()
        except Exception:
            pass  # Don't raise during GC

    def checkpoint(self) -> None:
        """Force a WAL checkpoint to flush pending writes to the main DB.

        Call this after bulk inserts to ensure data is durable even if
        the process restarts before the next auto-checkpoint.
        """
        with self._lock:
            if self._conn is not None:
                with contextlib.suppress(Exception):
                    self._conn.execute("PRAGMA wal_checkpoint(PASSIVE)")

    def close(self) -> None:
        """Close the database connection (for cleanup on shutdown).

        Performs a WAL checkpoint (truncating the WAL file) before
        closing to leave the database in a clean state.
        """
        with self._lock:
            if self._conn is not None:
                with contextlib.suppress(Exception):
                    self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                self._conn.close()
                self._conn = None
                _LOGGER.debug("Closed SQLite connection")

    # ------------------------------------------------------------------
    # Prediction operations
    # ------------------------------------------------------------------

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

    async def async_insert_prediction(
        self,
        start: str,
        price: float,
        confidence: float,
        hour: int,
        minute: int,
        stored_at: str,
    ) -> None:
        """Async wrapper for insert_prediction."""
        await self.hass.async_add_executor_job(
            self.insert_prediction,
            start,
            price,
            confidence,
            hour,
            minute,
            stored_at,
        )

    def delete_old_predictions(self, max_age_days: int) -> int:
        """Delete predictions older than max_age_days (blocking).

        Returns:
            Number of rows deleted.
        """
        cutoff = (datetime.now() - timedelta(days=max_age_days)).isoformat()
        with self._lock:
            conn = self._ensure_conn()
            cursor = conn.execute(
                "DELETE FROM predictions WHERE stored_at < ?", (cutoff,)
            )
            conn.commit()
            return cursor.rowcount

    async def async_delete_old_predictions(self, max_age_days: int) -> int:
        """Async wrapper for delete_old_predictions."""
        return await self.hass.async_add_executor_job(
            self.delete_old_predictions, max_age_days
        )

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

    async def async_find_predictions_for_timestamp(
        self, date_str: str, hour: int, minute: int
    ) -> list[dict[str, Any]]:
        """Async wrapper for find_predictions_for_timestamp."""
        return await self.hass.async_add_executor_job(
            self.find_predictions_for_timestamp, date_str, hour, minute
        )

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

    async def async_remove_prediction(self, prediction_id: int) -> None:
        """Async wrapper for remove_prediction."""
        await self.hass.async_add_executor_job(self.remove_prediction, prediction_id)

    def count_predictions(self) -> int:
        """Count total predictions in the database (blocking)."""
        with self._lock:
            conn = self._ensure_conn()
            row = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()
        return row[0] if row else 0

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
        cutoff = (datetime.now() - timedelta(days=max_age_days)).isoformat()
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
        cutoff = (datetime.now() - timedelta(days=max_age_days)).isoformat()
        with self._lock:
            conn = self._ensure_conn()
            cursor = conn.execute(
                "DELETE FROM nordpool_prognoses WHERE timestamp < ?", (cutoff,)
            )
            conn.commit()
            return cursor.rowcount

    def save_volatility(self, volatility: dict[int, float]) -> None:
        """Save per-slot volatility MAE values (blocking)."""
        with self._lock:
            conn = self._ensure_conn()
            for slot, value in volatility.items():
                conn.execute(
                    "INSERT OR REPLACE INTO volatility (slot, mae) VALUES (?, ?)",
                    (int(slot), float(value)),
                )
            conn.commit()

    def load_volatility(self) -> dict[int, float]:
        """Load per-slot volatility MAE values (blocking)."""
        with self._lock:
            conn = self._ensure_conn()
            rows = conn.execute("SELECT slot, mae FROM volatility").fetchall()
        return {row[0]: row[1] for row in rows}

    # ------------------------------------------------------------------
    # Error metrics operations
    # ------------------------------------------------------------------

    def save_error_metrics(self, error_metrics: dict[int, dict]) -> None:
        """Persist all error metrics to the database (blocking)."""
        with self._lock:
            conn = self._ensure_conn()
            for hour, metrics in error_metrics.items():
                data_json = json.dumps(
                    {
                        "errors": metrics.get("errors", []),
                        "abs_errors": metrics.get("abs_errors", []),
                        "pct_errors": metrics.get("pct_errors", []),
                        "predictions": metrics.get("predictions", []),
                        "actuals": metrics.get("actuals", []),
                        "count": metrics.get("count", 0),
                    }
                )
                conn.execute(
                    "INSERT OR REPLACE INTO error_metrics (hour, data) VALUES (?, ?)",
                    (int(hour), data_json),
                )
            conn.commit()

    async def async_save_error_metrics(self, error_metrics: dict[int, dict]) -> None:
        """Async wrapper for save_error_metrics."""
        await self.hass.async_add_executor_job(self.save_error_metrics, error_metrics)

    def load_error_metrics(self) -> dict[int, dict]:
        """Load all error metrics from the database (blocking)."""
        with self._lock:
            conn = self._ensure_conn()
            rows = conn.execute("SELECT hour, data FROM error_metrics").fetchall()
        result: dict[int, dict] = {}
        for hour, data_json in rows:
            metrics = json.loads(data_json)
            result[hour] = {
                "errors": metrics.get("errors", []),
                "abs_errors": metrics.get("abs_errors", []),
                "pct_errors": metrics.get("pct_errors", []),
                "predictions": metrics.get("predictions", []),
                "actuals": metrics.get("actuals", []),
                "count": metrics.get("count", 0),
            }
        return result

    async def async_load_error_metrics(self) -> dict[int, dict]:
        """Async wrapper for load_error_metrics."""
        return await self.hass.async_add_executor_job(self.load_error_metrics)

    # ------------------------------------------------------------------
    # Bias correction operations
    # ------------------------------------------------------------------

    def save_bias_correction(self, bias_correction: dict[int, float]) -> None:
        """Persist all bias corrections to the database (blocking)."""
        with self._lock:
            conn = self._ensure_conn()
            for hour, correction in bias_correction.items():
                conn.execute(
                    "INSERT OR REPLACE INTO bias_correction (hour, correction) VALUES (?, ?)",
                    (int(hour), float(correction)),
                )
            conn.commit()

    async def async_save_bias_correction(
        self, bias_correction: dict[int, float]
    ) -> None:
        """Async wrapper for save_bias_correction."""
        await self.hass.async_add_executor_job(
            self.save_bias_correction, bias_correction
        )

    def load_bias_correction(self) -> dict[int, float]:
        """Load all bias corrections from the database (blocking)."""
        with self._lock:
            conn = self._ensure_conn()
            rows = conn.execute(
                "SELECT hour, correction FROM bias_correction"
            ).fetchall()
        return {hour: correction for hour, correction in rows}

    async def async_load_bias_correction(self) -> dict[int, float]:
        """Async wrapper for load_bias_correction."""
        return await self.hass.async_add_executor_job(self.load_bias_correction)

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

    async def async_save_price_history(self, price_history: list[dict]) -> None:
        """Async wrapper for save_price_history."""
        await self.hass.async_add_executor_job(self.save_price_history, price_history)

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

    async def async_load_price_history(self) -> list[dict]:
        """Async wrapper for load_price_history."""
        return await self.hass.async_add_executor_job(self.load_price_history)

    # ------------------------------------------------------------------
    # Meta operations
    # ------------------------------------------------------------------

    def save_meta(self, training_samples: int, is_trained: bool) -> None:
        """Save metadata key/value pairs (blocking)."""
        with self._lock:
            conn = self._ensure_conn()
            conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                ("training_samples", str(training_samples)),
            )
            conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                ("is_trained", "1" if is_trained else "0"),
            )
            conn.commit()

    def save_meta_dict(self, extra_meta: dict) -> None:
        """Save arbitrary key/value pairs to the meta table (blocking).

        Used for persisting hyperparameter optimization results and
        other runtime-learned configuration.

        Args:
            extra_meta: Dict of string key -> string/int/float value
        """
        with self._lock:
            conn = self._ensure_conn()
            for key, value in extra_meta.items():
                conn.execute(
                    "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                    (str(key), str(value)),
                )
            conn.commit()

    def load_meta_dict(self) -> dict[str, str]:
        """Load all meta key/value pairs as strings (blocking)."""
        with self._lock:
            conn = self._ensure_conn()
            rows = conn.execute("SELECT key, value FROM meta").fetchall()
        return {row[0]: row[1] for row in rows}

    async def async_save_meta(self, training_samples: int, is_trained: bool) -> None:
        """Async wrapper for save_meta."""
        await self.hass.async_add_executor_job(
            self.save_meta, training_samples, is_trained
        )

    def load_meta(self) -> dict[str, Any]:
        """Load all metadata from the database (blocking)."""
        with self._lock:
            conn = self._ensure_conn()
            rows = conn.execute("SELECT key, value FROM meta").fetchall()
        meta = {}
        for key, value in rows:
            if key == "training_samples":
                meta["training_samples"] = int(value)
            elif key == "is_trained":
                meta["is_trained"] = value == "1"
            else:
                meta[key] = value
        return meta

    # ------------------------------------------------------------------
    # Bulk operations (backward-compatible)
    # ------------------------------------------------------------------

    def save_all(self, data: dict[str, Any]) -> None:
        """Save all learning data in one atomic transaction (blocking).

        All inserts happen within a single transaction. If any part
        fails, the entire save is rolled back.
        """
        with self._lock:
            conn = self._ensure_conn()
            try:
                # Error metrics
                error_metrics = data.get("error_metrics", {})
                for hour, metrics in error_metrics.items():
                    conn.execute(
                        "INSERT OR REPLACE INTO error_metrics (hour, data) VALUES (?, ?)",
                        (int(hour), json.dumps(metrics, default=str)),
                    )

                # Bias correction
                bias_correction = data.get("bias_correction", {})
                for hour, correction in bias_correction.items():
                    conn.execute(
                        "INSERT OR REPLACE INTO bias_correction (hour, correction) VALUES (?, ?)",
                        (int(hour), float(correction)),
                    )

                # Price history
                price_history = data.get("price_history", [])
                for entry in price_history:
                    date = entry.get("date")
                    prices = entry.get("prices", [])
                    if date:
                        conn.execute(
                            "INSERT OR REPLACE INTO price_history (date, prices) VALUES (?, ?)",
                            (date, json.dumps(prices)),
                        )

                # Predictions (backward compat — typically inserted individually)
                predictions = data.get("prediction_history", [])
                for p in predictions:
                    start = p.get("start", p.get("timestamp"))
                    price = p.get("price", p.get("predicted_price", 0))
                    if not start:
                        continue
                    conn.execute(
                        "INSERT INTO predictions "
                        "(start, price, confidence, hour, minute, stored_at) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            start,
                            float(price),
                            p.get("confidence", 0.6),
                            p.get("hour", 0),
                            p.get("minute", 0),
                            p.get("stored_at", ""),
                        ),
                    )

                # Volatility — ensure table exists (may be missing in DBs
                # created before the volatility feature was added)
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS volatility (slot INTEGER PRIMARY KEY, mae REAL NOT NULL)"
                )
                volatility = data.get("volatility_mae", {})
                for slot, value in volatility.items():
                    conn.execute(
                        "INSERT OR REPLACE INTO volatility (slot, mae) VALUES (?, ?)",
                        (int(slot), float(value)),
                    )

                # Solar scale
                solar_scale = data.get("solar_scale")
                if solar_scale is not None:
                    conn.execute(
                        "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                        ("solar_scale", str(solar_scale)),
                    )
                solar_scale_samples = data.get("solar_scale_samples")
                if solar_scale_samples is not None:
                    conn.execute(
                        "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                        ("solar_scale_samples", str(solar_scale_samples)),
                    )

                # Meta
                conn.execute(
                    "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                    ("training_samples", str(data.get("training_samples", 0))),
                )
                conn.execute(
                    "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                    (
                        "is_trained",
                        "1" if data.get("is_trained", False) else "0",
                    ),
                )

                conn.commit()

                # Count predictions passed vs total in DB for accurate log
                pred_count = conn.execute(
                    "SELECT COUNT(*) FROM predictions"
                ).fetchone()[0]

                _LOGGER.debug(
                    "Saved learning data (bias=%d, errors=%d, predictions=%d, price_days=%d, volatility=%d)",
                    len(bias_correction),
                    len(error_metrics),
                    pred_count,
                    len(price_history),
                    len(volatility),
                )
            except Exception:
                conn.rollback()
                _LOGGER.exception("Failed to save learning data — rolled back")
                raise

    async def async_save_all(self, data: dict[str, Any]) -> bool:
        """Async wrapper for save_all. Returns True on success."""
        try:
            await self.hass.async_add_executor_job(self.save_all, data)
            return True
        except Exception as err:
            _LOGGER.error("Failed to save learning data: %s", err)
            return False

    def load_all(self) -> dict[str, Any] | None:
        """Load all learning data from the database (blocking)."""
        with self._lock:
            conn = self._ensure_conn()

            # Check if tables exist
            table_check = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='predictions'"
            ).fetchone()
            if not table_check:
                return None

            # Error metrics — restore ALL keys (not just a hardcoded subset)
            error_metrics: dict[int, dict] = {}
            for hour, data_json in conn.execute(
                "SELECT hour, data FROM error_metrics"
            ).fetchall():
                metrics = json.loads(data_json)
                # Ensure numeric values are properly typed (json.loads may
                # produce string values if serialized with default=str)
                restored: dict[str, Any] = {}
                for k, v in metrics.items():
                    if isinstance(v, list):
                        restored[k] = [
                            float(x) if isinstance(x, (int, float, str)) else x
                            for x in v
                        ]
                    elif k == "count":
                        restored[k] = int(v)
                    else:
                        restored[k] = v
                error_metrics[hour] = restored

            # Bias correction
            bias_correction: dict[int, float] = {}
            for hour, correction in conn.execute(
                "SELECT hour, correction FROM bias_correction"
            ).fetchall():
                bias_correction[hour] = correction

            # Price history
            price_history: list[dict] = []
            for date, prices_json in conn.execute(
                "SELECT date, prices FROM price_history ORDER BY date ASC"
            ).fetchall():
                price_history.append({"date": date, "prices": json.loads(prices_json)})

            # Volatility
            volatility_mae: dict[int, float] = {}
            vol_table = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='volatility'"
            ).fetchone()
            if vol_table:
                for slot, mae in conn.execute(
                    "SELECT slot, mae FROM volatility"
                ).fetchall():
                    volatility_mae[slot] = mae

            # Meta
            meta = {}
            for key, value in conn.execute("SELECT key, value FROM meta").fetchall():
                if key == "training_samples":
                    meta["training_samples"] = int(value)
                elif key == "is_trained":
                    meta["is_trained"] = value == "1"
                else:
                    meta[key] = value

            pred_count = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]

        _LOGGER.info(
            "Loaded learning data (bias=%d, errors=%d, predictions=%d, price_days=%d, samples=%s, volatility=%d)",
            len(bias_correction),
            len(error_metrics),
            pred_count,
            len(price_history),
            meta.get("training_samples", 0),
            len(volatility_mae),
        )

        result: dict[str, Any] = {
            "bias_correction": bias_correction,
            "error_metrics": error_metrics,
            "prediction_count": pred_count,
            "price_history": price_history,
            "training_samples": meta.pop("training_samples", 0),
            "is_trained": meta.pop("is_trained", False),
            "volatility_mae": volatility_mae,
        }
        # Type-convert known meta keys that are stored as strings
        for key, converter in (
            ("solar_scale", float),
            ("solar_scale_samples", int),
        ):
            raw = meta.pop(key, None)
            if raw is not None:
                with contextlib.suppress(ValueError, TypeError):
                    result[key] = converter(raw)
        # Include remaining meta keys (HPO params, etc.)
        for key, value in meta.items():
            if key not in result and key not in ("schema_version",):
                result[key] = value
        return result

    async def async_load_all(self) -> dict[str, Any] | None:
        """Async wrapper for load_all."""
        return await self.hass.async_add_executor_job(self.load_all)

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def clear_all(self) -> None:
        """Delete all learning data (blocking).

        Drops tables and recreates the schema atomically —
        if recreation fails, the drop is rolled back.
        """
        with self._lock:
            conn = self._ensure_conn()
            try:
                conn.executescript(
                    """
                    DROP TABLE IF EXISTS predictions;
                    DROP TABLE IF EXISTS error_metrics;
                    DROP TABLE IF EXISTS bias_correction;
                    DROP TABLE IF EXISTS price_history;
                    DROP TABLE IF EXISTS volatility;
                    DROP TABLE IF EXISTS meta;
                    """
                )
                self._create_schema(conn)
                conn.commit()
            except Exception:
                conn.rollback()
                _LOGGER.exception("Failed to clear learning data — rolled back")
                raise

        _LOGGER.info("Cleared all learning data")

    async def async_clear_all(self) -> bool:
        """Async wrapper for clear_all. Returns True on success."""
        try:
            await self.hass.async_add_executor_job(self.clear_all)
            return True
        except Exception as err:
            _LOGGER.error("Failed to clear learning data: %s", err)
            return False

    # ------------------------------------------------------------------
    # Backward-compatible API (used by learning.py and predictor.py)
    # ------------------------------------------------------------------

    async def async_save_learning_data(self, data: dict[str, Any]) -> bool:
        """Save learning data (backward-compatible with old JSON API)."""
        return await self.async_save_all(data)

    async def async_load_learning_data(self) -> dict[str, Any] | None:
        """Load learning data (backward-compatible with old JSON API)."""
        return await self.async_load_all()

    async def async_clear_storage(self) -> bool:
        """Clear all learning data (backward-compatible with old JSON API)."""
        return await self.async_clear_all()
