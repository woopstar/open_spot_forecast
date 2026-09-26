"""The model's price history, stored per UTC slot (#24).

The ML layer keeps its price history as one entry per local day:
``{"date": "YYYY-MM-DD", "prices": [...]}``, one price per 15-minute slot
from local midnight (92/96/100 on DST days), None where the source had none.
It is stored as one ``spot_prices`` row per UTC slot start (``…Z``, the raw
spot price excl. VAT in the configured currency/kWh), so a 180-day history
is not rewritten as 180 JSON blobs, days never depend on how a list was cut,
and pruning and lookups compare UTC instants like every other table.

Schema v8 moves the JSON days of the old ``price_history`` table into
``spot_prices``; the empty legacy table only remains for older migrations.
These functions take an open connection: ``LearningStorage`` calls them
under its lock, inside its transactions.
"""

import json
import logging
import sqlite3
from collections.abc import Iterable
from datetime import date, datetime, timedelta
from typing import Any

from homeassistant.util import dt as dt_util

from ..price_series import same_prices
from ..time_slots import (
    UTC_KEY_FORMAT,
    local_midnight,
    parse_utc,
    slot_index_in_day,
    slot_start_in_day,
    slots_in_local_day,
    utc_slot_key,
)

_LOGGER = logging.getLogger(__name__)

PRICE_ROWS_SCHEMA_VERSION = 8

_IN_DAY = "julianday(timestamp) >= julianday(?) AND julianday(timestamp) < julianday(?)"


def _utc_key(moment: datetime) -> str:
    return moment.astimezone(dt_util.UTC).strftime(UTC_KEY_FORMAT)


def _day_bounds(day: date) -> tuple[str, str]:
    """Return a local day's UTC start and end keys."""
    return (
        _utc_key(local_midnight(day)),
        _utc_key(local_midnight(day + timedelta(days=1))),
    )


def day_rows(entry: dict[str, Any]) -> list[tuple[str, float]]:
    """Return a day entry's ``(utc slot key, price)`` rows; missing slots are skipped."""
    day = date.fromisoformat(str(entry.get("date")))
    return [
        (utc_slot_key(slot_start_in_day(day, index)), float(price))
        for index, price in enumerate(entry.get("prices") or [])
        if price is not None
    ]


def rows_to_days(rows: Iterable[tuple[str, float]]) -> list[dict[str, Any]]:
    """Return day entries, oldest first, from ``(utc slot key, price)`` rows.

    A day's list covers every slot of the local day (None where no row).
    """
    days: dict[date, list[float | None]] = {}
    for key, price in rows:
        moment = parse_utc(key)
        if moment is None:
            continue
        day = dt_util.as_local(moment).date()
        prices = days.setdefault(day, [None] * slots_in_local_day(day))
        index = slot_index_in_day(moment)
        if 0 <= index < len(prices):
            prices[index] = price
    return [
        {"date": day.isoformat(), "prices": prices}
        for day, prices in sorted(days.items())
    ]


def write_days(conn: sqlite3.Connection, entries: Iterable[dict[str, Any]]) -> int:
    """Replace the stored rows of each entry's day; return how many days."""
    written = 0
    for entry in entries:
        try:
            rows = day_rows(entry)
            bounds = _day_bounds(date.fromisoformat(str(entry.get("date"))))
        except TypeError, ValueError:
            continue
        conn.execute(f"DELETE FROM spot_prices WHERE {_IN_DAY}", bounds)  # noqa: S608
        conn.executemany(
            "INSERT OR REPLACE INTO spot_prices (timestamp, price) VALUES (?, ?)", rows
        )
        written += 1
    return written


def write_changed_days(
    conn: sqlite3.Connection,
    entries: Iterable[dict[str, Any]],
    saved: dict[str, list[float | None]],
) -> int:
    """Write the day entries that differ from ``saved``; return how many.

    ``saved`` holds each day's prices as last written or read, so a bulk save
    of a long history only rewrites the days that changed.
    """
    changed = [
        entry
        for entry in entries
        if entry.get("date")
        and not same_prices(
            saved.get(str(entry["date"]), []), entry.get("prices") or []
        )
    ]
    written = write_days(conn, changed)
    for entry in changed:
        saved[str(entry["date"])] = list(entry.get("prices") or [])
    return written


def read_days(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Return every stored day entry, oldest first."""
    rows = conn.execute(
        "SELECT timestamp, price FROM spot_prices ORDER BY julianday(timestamp)"
    ).fetchall()
    return rows_to_days(rows)


def delete_days_before(conn: sqlite3.Connection, before: date) -> int:
    """Delete the rows of the local days before ``before``; return how many rows."""
    cursor = conn.execute(
        "DELETE FROM spot_prices WHERE julianday(timestamp) < julianday(?)",
        (_utc_key(local_midnight(before)),),
    )
    return cursor.rowcount


def migrate_price_history_to_rows(conn: sqlite3.Connection) -> None:
    """Move the JSON days of ``price_history`` into ``spot_prices`` (schema < 8).

    Args:
        conn: Open learning database connection with every table created;
            the caller commits.
    """
    row = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
    version = int(row[0]) if row else 0
    if version >= PRICE_ROWS_SCHEMA_VERSION:
        return
    entries = []
    for day, prices_json in conn.execute("SELECT date, prices FROM price_history"):
        try:
            entries.append({"date": day, "prices": json.loads(prices_json)})
        except TypeError, ValueError:
            continue
    moved = write_days(conn, entries)
    conn.execute("DELETE FROM price_history")
    conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', ?)",
        (str(PRICE_ROWS_SCHEMA_VERSION),),
    )
    if moved:
        _LOGGER.info("Stored %d days of price history per UTC slot", moved)
