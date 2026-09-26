"""Storage module for persisting ML learning data using SQLite.

``LearningStorage`` owns the connection, the write lock, the schema and its
migrations. The table operations live in mixins, one module per group of
tables: ``prediction_storage.py``, ``history_storage.py``,
``series_storage.py`` (time-series source tables), ``state_storage.py`` and
``accuracy_storage.py``.
"""

import contextlib
import json
import logging
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

from homeassistant.core import HomeAssistant

from .accuracy_storage import LeadTimeAccuracyStorageMixin
from .bias_storage import migrate_bias_to_additive
from .history_storage import HistoryStorageMixin
from .prediction_storage import PredictionStorageMixin
from .price_storage import migrate_price_history_to_rows
from .series_storage import SeriesStorageMixin
from .spot_migration import migrate_to_spot_prices
from .state_storage import LearningStateStorageMixin
from .weather_migration import migrate_weather_to_utc

_LOGGER = logging.getLogger(__name__)


class LearningStorage(
    PredictionStorageMixin,
    HistoryStorageMixin,
    SeriesStorageMixin,
    LearningStateStorageMixin,
    LeadTimeAccuracyStorageMixin,
):
    """Handles persistence of learning data to a SQLite database.

    Uses a single persistent connection with WAL journal mode.
    All write operations are serialized via a threading lock.
    The database file is created automatically on first use.

    Schema (and the mixin that reads and writes it):
      predictions     — per-interval forecasts awaiting self-learning
                        comparison (prediction_storage.py)
      error_metrics   — per-hour error tracking (JSON-serialized arrays)
      bias_correction — per-slot additive bias offsets (0-95)
      volatility      — per-slot volatility MAE
      meta            — key/value pairs (training_samples, is_trained,
                        hpo_counter) (all four: state_storage.py)
      spot_prices     — the model's price history per UTC 15-min slot (#24;
                        price_storage.py); price_history is its legacy,
                        emptied JSON-per-day form
      weather_history — 15-min weather snapshots, keyed by UTC slot start
      nordpool_prognoses — hourly Nordpool prognoses (all three:
                        history_storage.py; kept current by the Nordpool
                        time-series source through series_storage.py)
      dayahead_prices — raw day-ahead prices, EUR/MWh per UTC 15-min slot
                        (series_storage.py; energy-charts / ENTSO-E source)
      openmeteo_weather — Open-Meteo weather per sampling point and UTC
                        15-min slot (series_storage.py; Open-Meteo source)
      lead_time_accuracy — daily per-lead-time error sums (accuracy_storage.py)
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
        self._saved_price_days: dict[str, list[float | None]] = {}

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
        and the old JSON learning file exists, imports its training state
        and renames the JSON file. Its prices, predictions, error metrics and
        bias factors are not imported (see migrate_to_spot_prices).

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

            # Legacy prices, predictions and errors are consumer prices; the
            # model learns the raw spot price (#16). Legacy bias factors are
            # multiplicative; bias offsets are additive (#15). None is imported.
            predictions: list = []
            error_metrics: dict = {}
            bias_correction: dict = {}
            price_history: list = []
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

            CREATE TABLE IF NOT EXISTS spot_prices (
                timestamp       TEXT    PRIMARY KEY,
                price           REAL    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS dayahead_prices (
                timestamp       TEXT    PRIMARY KEY,
                price           REAL
            );

            CREATE TABLE IF NOT EXISTS openmeteo_weather (
                timestamp       TEXT    NOT NULL,
                point           TEXT    NOT NULL,
                wind_80m        REAL,
                temperature     REAL,
                irradiance      REAL,
                pressure        REAL,
                humidity        REAL,
                PRIMARY KEY (timestamp, point)
            );

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

        migrate_bias_to_additive(conn)
        self._create_lead_time_accuracy_schema(conn)
        migrate_to_spot_prices(conn)
        migrate_weather_to_utc(conn)
        migrate_price_history_to_rows(conn)
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
                    DROP TABLE IF EXISTS spot_prices;
                    DROP TABLE IF EXISTS volatility;
                    DROP TABLE IF EXISTS lead_time_accuracy;
                    DROP TABLE IF EXISTS meta;
                    """
                )
                self._create_schema(conn)
                conn.commit()
                self._saved_price_days = {}
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
    # Backward-compatible API (used by learning.py)
    # ------------------------------------------------------------------

    async def async_clear_storage(self) -> bool:
        """Clear all learning data (backward-compatible with old JSON API)."""
        return await self.async_clear_all()
