"""Schema v7: weather snapshots are keyed by their UTC slot start (#59).

Snapshots used to be stored with ``datetime.now().isoformat()`` (naive, in
the container's time zone) and later with ``dt_util.now().isoformat()``
(local time with its offset and microseconds). Every row is rewritten once
to the one stored format, the UTC start of its 15-minute slot
(``utc_slot_key``, e.g. ``2026-09-24T08:00:00Z``), so lookups and pruning
compare like with like. A naive timestamp is read as Home Assistant local
time; one in the repeated hour of a DST fall-back day is ambiguous and is
taken as the first pass. When several snapshots fall into one slot the
earliest is kept, as training does (``TrainingInputs``).
"""

import logging
import sqlite3
from datetime import datetime

from ..time_slots import parse_utc, utc_slot_key

_LOGGER = logging.getLogger(__name__)

WEATHER_UTC_SCHEMA_VERSION = 7


def migrate_weather_to_utc(conn: sqlite3.Connection) -> None:
    """Rewrite weather snapshot timestamps as UTC slot keys (schema version < 7).

    Args:
        conn: Open learning database connection with every table created;
            the caller commits.
    """
    row = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
    version = int(row[0]) if row else 0
    if version >= WEATHER_UTC_SCHEMA_VERSION:
        return

    rows = conn.execute(
        """SELECT timestamp, temperature, wind_speed, wind_direction,
                  cloud_coverage, humidity, solar_power
           FROM weather_history"""
    ).fetchall()
    snapshots: dict[str, tuple[datetime, tuple]] = {}
    unreadable = 0
    for timestamp, *values in rows:
        moment = parse_utc(timestamp)
        if moment is None:
            unreadable += 1
            continue
        key = utc_slot_key(moment)
        kept = snapshots.get(key)
        if kept is None or moment < kept[0]:
            snapshots[key] = (moment, tuple(values))

    conn.execute("DELETE FROM weather_history")
    conn.executemany(
        """INSERT INTO weather_history
           (timestamp, temperature, wind_speed, wind_direction,
            cloud_coverage, humidity, solar_power)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        [(key, *values) for key, (_moment, values) in snapshots.items()],
    )
    conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', ?)",
        (str(WEATHER_UTC_SCHEMA_VERSION),),
    )
    if rows:
        _LOGGER.info(
            "Weather snapshots are now keyed by their UTC slot start: rewrote %d, "
            "merged %d into an earlier snapshot of the same slot, dropped %d "
            "unreadable",
            len(snapshots),
            len(rows) - unreadable - len(snapshots),
            unreadable,
        )
