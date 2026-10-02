"""SQLite persistence for the predictor's learned state.

Mixed into ``LearningStorage`` so the learning database keeps a single
connection and write lock. The bulk ``save_all`` / ``load_all`` the predictor
uses persist and restore the per-slot ``error_metrics``, the ``bias_correction``
offsets per slot and lead-time bucket (#118), ``volatility`` and the ``meta``
key/value table in one transaction; ``save_meta_dict`` / ``delete_meta_keys``
write individual ``meta`` keys (holdout metrics, hyperparameters, source state).
"""

import contextlib
import json
import logging
from typing import Any

from .price_storage import read_days, write_changed_days
from .storage_base import StorageMixinBase

_LOGGER = logging.getLogger(__name__)

_BIAS_INSERT = (
    "INSERT OR REPLACE INTO bias_correction (hour, bucket, correction) VALUES (?, ?, ?)"
)
_BIAS_SELECT = "SELECT hour, bucket, correction FROM bias_correction"


def bias_rows(
    bias_correction: dict[int, dict[str, float]],
) -> list[tuple[int, str, float]]:
    """Return ``(slot, bucket, offset)`` rows of nested bias offsets.

    Raises:
        TypeError, ValueError: A slot, bucket or offset of the wrong type.
    """
    return [
        (int(slot), str(bucket), float(offset))
        for slot, offsets in bias_correction.items()
        for bucket, offset in offsets.items()
    ]


def nest_bias_rows(rows: list[tuple[int, str, float]]) -> dict[int, dict[str, float]]:
    """Return ``{slot: {bucket: offset}}`` from ``(slot, bucket, offset)`` rows."""
    bias_correction: dict[int, dict[str, float]] = {}
    for slot, bucket, offset in rows:
        bias_correction.setdefault(slot, {})[bucket] = offset
    return bias_correction


class LearningStateStorageMixin(StorageMixinBase):
    """Bulk learned-state I/O and ``meta`` key writes for ``LearningStorage``."""

    # ------------------------------------------------------------------
    # Meta operations
    # ------------------------------------------------------------------

    def save_meta_dict(self, extra_meta: dict) -> None:
        """Save arbitrary key/value pairs to the meta table (blocking).

        Used for persisting the holdout metrics, source state and other
        runtime-learned configuration.

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

    def delete_meta_keys(self, keys: tuple[str, ...]) -> None:
        """Delete key/value pairs from the meta table (blocking)."""
        with self._lock:
            conn = self._ensure_conn()
            conn.executemany("DELETE FROM meta WHERE key = ?", [(k,) for k in keys])
            conn.commit()

    # ------------------------------------------------------------------
    # Bulk operations
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

                # Bias correction, per slot and lead-time bucket
                bias_correction = data.get("bias_correction", {})
                conn.executemany(_BIAS_INSERT, bias_rows(bias_correction))

                # Price history: only the days that changed (#24)
                price_history = data.get("price_history", [])
                write_changed_days(conn, price_history, self._saved_price_days)

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

            # Bias correction, per slot and lead-time bucket
            bias_correction = nest_bias_rows(conn.execute(_BIAS_SELECT).fetchall())

            # Price history, stored per UTC slot (#24)
            price_history = read_days(conn)
            self._saved_price_days = {
                day["date"]: list(day["prices"]) for day in price_history
            }

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
        # Include remaining meta keys (holdout metrics, source state, etc.)
        for key, value in meta.items():
            if key not in result and key not in ("schema_version",):
                result[key] = value
        return result

    async def async_load_all(self) -> dict[str, Any] | None:
        """Async wrapper for load_all."""
        return await self.hass.async_add_executor_job(self.load_all)
