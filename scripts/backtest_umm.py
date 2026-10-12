"""Nord Pool UMM outage messages for the backtest (dev-only, #123).

The UMM API keeps every message version back to 2015, so the backtest can
rebuild what the market knew at any origin: the messages whose events
overlap the test period are fetched one calendar month at a time (every
version, paged), parsed with the integration's ``parse_umm_messages`` and
indexed by ``OutageIndex``. Complete past months are cached as parsed rows
in the cache directory. Training rows get the outages published by their
day-ahead gate and target rows those published before the horizon cutoff,
so no row sees a message the market did not have (``feature_matrix`` in
``scripts/backtest.py``).

``load_outages_db`` reads the same rows from a learning-database export
instead (``--outages-db``, #168): the archive a live instance stored, with
every version it saw. That is the only complete history for the zones on
the ENTSO-E platform, whose API serves a document's latest revision only.
"""

import json
import sqlite3
import sys
import urllib.parse
from collections.abc import Callable, Iterator
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from custom_components.open_spot_forecast.api.nordpool_umm import (
    parse_umm_messages,
    umm_query,
)
from custom_components.open_spot_forecast.const import NORDPOOL_UMM_API, REGIONS
from custom_components.open_spot_forecast.ml.outages import (
    KIND_PRODUCTION,
    OUTAGE_FEATURES,
    OutageIndex,
)

# Every stored version with its periods; a version without one comes back
# once with NULL period columns, so it still supersedes the earlier ones
_OUTAGE_DB_SQL = (
    "SELECT m.message_id, m.version, m.published, m.status, p.unit, p.kind, "
    "p.event_start, p.event_stop, p.unavailable_mw "
    "FROM umm_messages m LEFT JOIN umm_periods p "
    "ON p.message_id = m.message_id AND p.version = m.version "
    "ORDER BY m.published, m.message_id, m.version"
)
_OUTAGE_DB_COLUMNS = (
    "message_id",
    "version",
    "published",
    "status",
    "unit",
    "kind",
    "event_start",
    "event_stop",
    "unavailable_mw",
)


class PlantOutageIndex(OutageIndex):
    """An ``OutageIndex`` without the grid feature: plants only (#168).

    ``unavailable_transmission`` is left out of a slot's inputs, so it is
    unknown (NaN) in every row, as in a region without an outage source.
    """

    def for_slot(self, start: datetime, origin: datetime) -> dict[str, float]:
        """Return the slot's unavailable production as known at ``origin``."""
        feature = OUTAGE_FEATURES[KIND_PRODUCTION]
        return {feature: super().for_slot(start, origin)[feature]}


def months(first: date, last: date) -> Iterator[tuple[date, date]]:
    """Yield (first day, first day of the next month) for every month touched."""
    current = first.replace(day=1)
    while current <= last:
        following = (current.replace(day=28) + timedelta(days=4)).replace(day=1)
        yield current, following
        current = following


def fetch_umm_month(
    eic: str,
    month_start: date,
    month_end: date,
    fetch: Callable[[str], dict[str, Any]],
    published_until: datetime,
) -> list[dict[str, Any]]:
    """Return the parsed rows of every message version with an event in the month."""
    start = datetime.combine(month_start, datetime.min.time(), UTC)
    end = datetime.combine(month_end, datetime.min.time(), UTC)
    rows: list[dict[str, Any]] = []
    skip = 0
    while True:
        query = urllib.parse.urlencode(
            umm_query(eic, start, end, published_until, skip=skip)
        )
        payload = fetch(f"{NORDPOOL_UMM_API}?{query}")
        rows.extend(parse_umm_messages(payload, eic))
        items = payload.get("items") or []
        skip += len(items)
        if not items or skip >= int(payload.get("total") or 0):
            return rows


def load_umm_outages(
    region: str,
    first: date,
    last: date,
    cache_dir: Path,
    fetch: Callable[[str], dict[str, Any]],
    today: date | None = None,
    index: type[OutageIndex] = OutageIndex,
) -> OutageIndex:
    """Load the region's UMM outages for events on the UTC days ``first``..``last``.

    One request series per calendar month (a long outage appears in each
    month it touches; the index counts a period once). Months that ended
    before ``today`` are cached as parsed rows in ``cache_dir``. ``index``
    is the index class the rows go into.
    """
    eic = str(REGIONS[region]["entsoe"])
    today = today or date.today()
    now = datetime.now(UTC)
    rows: list[dict[str, Any]] = []
    for month_start, month_end in months(first, last):
        cache_file = cache_dir / f"umm_{region}_{month_start:%Y-%m}.json"
        if cache_file.exists():
            month = json.loads(cache_file.read_text(encoding="utf-8"))
        else:
            print(f"  UMM {region} {month_start:%Y-%m} ...", file=sys.stderr)
            month = fetch_umm_month(eic, month_start, month_end, fetch, now)
            if month_end < today:
                cache_dir.mkdir(parents=True, exist_ok=True)
                cache_file.write_text(json.dumps(month), encoding="utf-8")
        rows.extend(month)
    return index(rows)


def load_outages_db(path: Path, index: type[OutageIndex] = OutageIndex) -> OutageIndex:
    """Index every outage message version stored in a learning-database export.

    The export (``umm_messages`` / ``umm_periods``, filled by the Nord Pool
    UMM or the ENTSO-E outage source) is opened read-only and immutable: the
    backtest never migrates, checkpoints or writes it. The index gives a
    slot what the instance had stored by an origin, so a version it fetched
    later, or a later revision, never reaches an earlier origin.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    if not path.is_file():
        raise FileNotFoundError(path)
    connection = sqlite3.connect(
        f"{path.resolve().as_uri()}?mode=ro&immutable=1", uri=True
    )
    try:
        rows = connection.execute(_OUTAGE_DB_SQL).fetchall()
    finally:
        connection.close()
    return index(dict(zip(_OUTAGE_DB_COLUMNS, row, strict=True)) for row in rows)
