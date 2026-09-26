"""Generic SQLite access for time-series source tables (#32).

A ``SeriesSpec`` describes a table of rows on a fixed UTC grid: one row per
``timestamp`` (``…Z``), or per timestamp and key (e.g. a weather sampling
point), with value columns. ``SeriesStorageMixin`` gives every such table
the same operations: which grid points are stored (``series_timestamps``),
a change-detecting upsert, range loads and retention pruning. A source's
``SourceState`` (see ``time_series.py``) is kept in ``meta``.

Table and column names come from the specs defined in code, never from
input, so building the SQL from them is safe.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from homeassistant.util import dt as dt_util

from ..time_slots import UTC_KEY_FORMAT, parse_utc
from .storage_base import StorageMixinBase

_STATE_KEY_PREFIX = "source_state_"


@dataclass(frozen=True, slots=True)
class SeriesSpec:
    """A time-series table: its grid, key and value columns.

    A grid point counts as stored when its row has a value in every column
    (for a keyed table: a complete row for every expected key).
    """

    name: str
    table: str
    columns: tuple[str, ...]
    step_minutes: int
    key_column: str | None = None

    @property
    def conflict_columns(self) -> str:
        """Return the columns that identify a row."""
        return f"timestamp, {self.key_column}" if self.key_column else "timestamp"


# Nordpool's consumption and production prognoses, one row per UTC hour
NORDPOOL_PROGNOSES = SeriesSpec(
    name="nordpool",
    table="nordpool_prognoses",
    columns=("consumption", "solar", "wind_offshore", "wind_onshore"),
    step_minutes=60,
)

# Day-ahead auction prices (EUR/MWh, raw: no VAT or tariffs), per 15-min slot
DAYAHEAD_PRICES = SeriesSpec(
    name="dayahead",
    table="dayahead_prices",
    columns=("price",),
    step_minutes=15,
)

# Open-Meteo weather per sampling point ("lat,lon") and 15-min slot (#22)
OPENMETEO_WEATHER = SeriesSpec(
    name="openmeteo",
    table="openmeteo_weather",
    columns=("wind_80m", "temperature", "irradiance", "pressure", "humidity"),
    step_minutes=15,
    key_column="point",
)


def _utc_key(moment: datetime) -> str:
    """Return a datetime as the stored ``…Z`` key."""
    return moment.astimezone(UTC).strftime(UTC_KEY_FORMAT)


def _in_range_sql() -> str:
    return (
        "julianday(timestamp) >= julianday(?) AND julianday(timestamp) < julianday(?)"
    )


class SeriesStorageMixin(StorageMixinBase):
    """Time-series table operations for ``LearningStorage``."""

    def series_timestamps(
        self,
        spec: SeriesSpec,
        start: datetime,
        end: datetime,
        keys: Sequence[str] = (),
    ) -> list[datetime]:
        """Return the stored grid points in ``[start, end)`` (blocking).

        Args:
            spec: The table.
            start: Range start.
            end: Range end (exclusive).
            keys: For a keyed table, the keys every point needs a row for.
        """
        complete = " AND ".join(f"{column} IS NOT NULL" for column in spec.columns)
        params: list[Any] = [_utc_key(start), _utc_key(end)]
        sql = (
            f"SELECT timestamp FROM {spec.table} WHERE {_in_range_sql()} AND {complete}"  # noqa: S608
        )
        if spec.key_column:
            placeholders = ", ".join("?" for _ in keys)
            sql += (
                f" AND {spec.key_column} IN ({placeholders})"
                f" GROUP BY timestamp HAVING COUNT(DISTINCT {spec.key_column}) = ?"
            )
            params.extend([*keys, len(set(keys))])
        with self._lock:
            rows = self._ensure_conn().execute(sql, params).fetchall()
        return [moment for (value,) in rows if (moment := parse_utc(value)) is not None]

    def upsert_series(self, spec: SeriesSpec, rows: Iterable[dict[str, Any]]) -> bool:
        """Insert or update rows; return whether any stored value changed (blocking).

        New values replace stored ones; a missing (None) value keeps the
        stored one, so a partial response never erases data. Timestamps are
        stored as UTC ``…Z`` keys. A change moves ``last_data_write``, which
        makes the predictor retrain.
        """
        key = [spec.key_column] if spec.key_column else []
        names = ["timestamp", *key, *spec.columns]
        values = []
        for row in rows:
            moment = parse_utc(row.get("timestamp"))
            if moment is None:
                continue
            values.append([_utc_key(moment), *(row.get(name) for name in names[1:])])
        if not values:
            return False
        updates = ", ".join(f"{c} = COALESCE(excluded.{c}, {c})" for c in spec.columns)
        differs = " OR ".join(
            f"COALESCE(excluded.{c}, {c}) IS NOT {c}" for c in spec.columns
        )
        sql = (
            f"INSERT INTO {spec.table} ({', '.join(names)}) "  # noqa: S608
            f"VALUES ({', '.join('?' for _ in names)}) "
            f"ON CONFLICT({spec.conflict_columns}) DO UPDATE SET {updates} "
            f"WHERE {differs}"
        )
        with self._lock:
            conn = self._ensure_conn()
            changed = conn.executemany(sql, values).rowcount > 0
            conn.commit()
            if changed:
                self.last_data_write = dt_util.utcnow()
        return changed

    def load_series(
        self, spec: SeriesSpec, start: datetime, end: datetime
    ) -> list[dict[str, Any]]:
        """Return the rows in ``[start, end)``, oldest first (blocking)."""
        key = [spec.key_column] if spec.key_column else []
        names = ["timestamp", *key, *spec.columns]
        sql = (
            f"SELECT {', '.join(names)} FROM {spec.table} "  # noqa: S608
            f"WHERE {_in_range_sql()} ORDER BY julianday(timestamp)"
        )
        with self._lock:
            rows = (
                self._ensure_conn()
                .execute(sql, (_utc_key(start), _utc_key(end)))
                .fetchall()
            )
        return [dict(zip(names, row, strict=True)) for row in rows]

    def prune_series(self, spec: SeriesSpec, before: datetime) -> int:
        """Delete rows older than ``before``; return how many (blocking)."""
        with self._lock:
            conn = self._ensure_conn()
            cursor = conn.execute(
                f"DELETE FROM {spec.table} "  # noqa: S608
                "WHERE julianday(timestamp) < julianday(?)",
                (_utc_key(before),),
            )
            conn.commit()
        return cursor.rowcount

    def load_source_state(self, name: str) -> str | None:
        """Return a source's stored state (``SourceState`` JSON) (blocking)."""
        with self._lock:
            row = (
                self._ensure_conn()
                .execute(
                    "SELECT value FROM meta WHERE key = ?", (_STATE_KEY_PREFIX + name,)
                )
                .fetchone()
            )
        return row[0] if row else None

    def save_source_state(self, name: str, value: str) -> None:
        """Store a source's state (``SourceState`` JSON) (blocking)."""
        with self._lock:
            conn = self._ensure_conn()
            conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                (_STATE_KEY_PREFIX + name, value),
            )
            conn.commit()
