"""SQLite access for the stored Nord Pool UMM outage messages (#123).

Two tables, both written by ``NordpoolUmmSource`` (``api/nordpool_umm.py``)
and read by ``OutageIndex`` (``ml/outages.py``):

* ``umm_messages`` — every **version** of every message that named a unit
  in the region's area: when it was published and its status. A version
  never changes once published, so rows are only ever added. Versions that
  drop the region's units are kept too: they supersede the earlier ones.
* ``umm_periods`` — the version's periods of unavailable capacity for the
  region's units (production/generation units in the area, transmission
  units into or out of it).

Both tables are created with ``CREATE TABLE IF NOT EXISTS``. The source's
coverage is kept in ``meta`` under ``source_state_umm_outages`` (JSON).
"""

from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

from homeassistant.util import dt as dt_util

from ..time_slots import UTC_KEY_FORMAT
from .storage_base import StorageMixinBase

UMM_SCHEMA_SQL = """
    CREATE TABLE IF NOT EXISTS umm_messages (
        message_id          TEXT    NOT NULL,
        version             INTEGER NOT NULL,
        published           TEXT    NOT NULL,
        message_type        INTEGER NOT NULL,
        unavailability_type INTEGER,
        status              INTEGER NOT NULL,
        PRIMARY KEY (message_id, version)
    );

    CREATE TABLE IF NOT EXISTS umm_periods (
        message_id      TEXT    NOT NULL,
        version         INTEGER NOT NULL,
        unit            TEXT    NOT NULL,
        kind            TEXT    NOT NULL,
        fuel_type       INTEGER,
        event_start     TEXT    NOT NULL,
        event_stop      TEXT    NOT NULL,
        unavailable_mw  REAL    NOT NULL,
        installed_mw    REAL,
        PRIMARY KEY (message_id, version, unit, event_start, event_stop)
    );

    CREATE INDEX IF NOT EXISTS idx_umm_periods_stop
        ON umm_periods(event_stop);
"""

_MESSAGE_COLUMNS = (
    "message_id",
    "version",
    "published",
    "message_type",
    "unavailability_type",
    "status",
)
_PERIOD_COLUMNS = (
    "message_id",
    "version",
    "unit",
    "kind",
    "fuel_type",
    "event_start",
    "event_stop",
    "unavailable_mw",
    "installed_mw",
)


def _utc_key(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime(UTC_KEY_FORMAT)


class OutageStorageMixin(StorageMixinBase):
    """``umm_messages`` and ``umm_periods`` operations for ``LearningStorage``."""

    def upsert_umm_rows(self, rows: Iterable[dict[str, Any]]) -> bool:
        """Store parsed UMM rows; return whether a new version was added (blocking).

        Rows are the flat form ``parse_umm_messages`` returns: one per
        period, one per version without a period for the region (``kind``
        None). A version already stored is left alone. A new version moves
        ``last_data_write``, so the model retrains on it.
        """
        messages: dict[tuple[str, int], list[Any]] = {}
        periods: list[list[Any]] = []
        for row in rows:
            key = (str(row["message_id"]), int(row["version"]))
            messages.setdefault(key, [row.get(name) for name in _MESSAGE_COLUMNS])
            if row.get("kind") is not None:
                periods.append([row.get(name) for name in _PERIOD_COLUMNS])
        if not messages:
            return False
        with self._lock:
            conn = self._ensure_conn()
            added = conn.executemany(
                f"INSERT OR IGNORE INTO umm_messages ({', '.join(_MESSAGE_COLUMNS)}) "  # noqa: S608
                f"VALUES ({', '.join('?' for _ in _MESSAGE_COLUMNS)})",
                list(messages.values()),
            ).rowcount
            conn.executemany(
                f"INSERT OR IGNORE INTO umm_periods ({', '.join(_PERIOD_COLUMNS)}) "  # noqa: S608
                f"VALUES ({', '.join('?' for _ in _PERIOD_COLUMNS)})",
                periods,
            )
            conn.commit()
            changed = added > 0
            if changed:
                self.last_data_write = dt_util.utcnow()
        return changed

    def load_umm_rows(self, start: datetime, end: datetime) -> list[dict[str, Any]]:
        """Return every stored version, with its periods overlapping ``[start, end)``.

        Every version comes back at least once (``kind`` None without a
        period in the range), so ``OutageIndex`` can tell which version of
        a message was current at an origin even when that version has no
        period in the range (blocking).
        """
        names = [*_MESSAGE_COLUMNS, *_PERIOD_COLUMNS[2:]]
        sql = (
            "SELECT m.message_id, m.version, m.published, m.message_type, "
            "m.unavailability_type, m.status, p.unit, p.kind, p.fuel_type, "
            "p.event_start, p.event_stop, p.unavailable_mw, p.installed_mw "
            "FROM umm_messages m LEFT JOIN umm_periods p "
            "ON p.message_id = m.message_id AND p.version = m.version "
            "AND julianday(p.event_stop) > julianday(?) "
            "AND julianday(p.event_start) < julianday(?) "
            "ORDER BY m.published, m.message_id, m.version"
        )
        with self._lock:
            rows = (
                self._ensure_conn()
                .execute(sql, (_utc_key(start), _utc_key(end)))
                .fetchall()
            )
        return [dict(zip(names, row, strict=True)) for row in rows]

    def prune_umm_rows(self, before: datetime) -> int:
        """Delete periods ending before ``before`` and versions left without any.

        A version published before ``before`` whose periods are all gone
        is deleted; later versions stay (blocking).

        Returns:
            The number of message versions deleted.
        """
        key = _utc_key(before)
        with self._lock:
            conn = self._ensure_conn()
            conn.execute(
                "DELETE FROM umm_periods WHERE julianday(event_stop) < julianday(?)",
                (key,),
            )
            cursor = conn.execute(
                "DELETE FROM umm_messages WHERE julianday(published) < julianday(?) "
                "AND NOT EXISTS (SELECT 1 FROM umm_periods p "
                "WHERE p.message_id = umm_messages.message_id "
                "AND p.version = umm_messages.version)",
                (key,),
            )
            conn.commit()
        return cursor.rowcount
