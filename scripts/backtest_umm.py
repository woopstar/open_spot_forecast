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
"""

import json
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
from custom_components.open_spot_forecast.ml.outages import OutageIndex


def _months(first: date, last: date) -> Iterator[tuple[date, date]]:
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
) -> OutageIndex:
    """Load the region's UMM outages for events on the UTC days ``first``..``last``.

    One request series per calendar month (a long outage appears in each
    month it touches; the index counts a period once). Months that ended
    before ``today`` are cached as parsed rows in ``cache_dir``.
    """
    eic = str(REGIONS[region]["entsoe"])
    today = today or date.today()
    now = datetime.now(UTC)
    rows: list[dict[str, Any]] = []
    for month_start, month_end in _months(first, last):
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
    return OutageIndex(rows)
