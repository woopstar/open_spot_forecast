"""ENTSO-E outage documents for the backtest (dev-only, #138).

The Transparency Platform serves the current revision of every outage
document (A77/A80 per bidding zone, A78 per border and direction), so the
backtest rebuilds what the market knew at an origin from each document's
latest revision and its publication time — the view a fresh install has
(``api/entsoe_outages.py``). The documents whose events overlap the test
period are fetched one calendar month at a time, paged with ``offset`` and
halved when a month exceeds the offset limit, parsed with the
integration's ``parse_entsoe_outages`` and indexed by ``OutageIndex``.
Complete past months are cached as parsed rows in the cache directory; the
token is never cached or printed. Training rows get the documents published
by their day-ahead gate and target rows those published before the horizon
cutoff (``feature_matrix`` in ``scripts/backtest.py``).
"""

import json
import sys
import urllib.parse
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from custom_components.open_spot_forecast.api.entsoe import entsoe_period
from custom_components.open_spot_forecast.api.entsoe_outages import (
    ENTSOE_DOCUMENT_LIMIT,
    ENTSOE_OFFSET_LIMIT,
    entsoe_outage_queries,
    parse_entsoe_outages,
)
from custom_components.open_spot_forecast.const import (
    ENTSOE_API,
    ENTSOE_OUTAGE_BORDERS,
    REGIONS,
)
from custom_components.open_spot_forecast.ml.outages import OutageIndex

from .backtest_umm import months

# A window this short is not halved further when it still overflows
_MIN_WINDOW = timedelta(days=1)


def fetch_outage_query(
    query: dict[str, str],
    start: datetime,
    end: datetime,
    api_key: str,
    fetch: Callable[[str], bytes],
) -> list[dict[str, Any]]:
    """Return the rows of one query over ``[start, end)``, every page.

    A window with more documents than the offset limit allows is halved.
    """
    rows: list[dict[str, Any]] = []
    for offset in range(0, ENTSOE_OFFSET_LIMIT + 1, ENTSOE_DOCUMENT_LIMIT):
        params = query | entsoe_period(start, end) | {"offset": str(offset)}
        encoded = urllib.parse.urlencode(params | {"securityToken": api_key})
        response = parse_entsoe_outages(fetch(f"{ENTSOE_API}?{encoded}"))
        rows.extend(response.rows)
        if response.documents < ENTSOE_DOCUMENT_LIMIT:
            return rows
    if end - start <= _MIN_WINDOW:
        print(
            f"  {query['documentType']}: too many documents, rest skipped",
            file=sys.stderr,
        )
        return rows
    middle = start + (end - start) / 2
    return fetch_outage_query(
        query, start, middle, api_key, fetch
    ) + fetch_outage_query(query, middle, end, api_key, fetch)


def fetch_entsoe_outage_month(
    region: str,
    month_start: date,
    month_end: date,
    api_key: str,
    fetch: Callable[[str], bytes],
) -> list[dict[str, Any]]:
    """Return the parsed rows of every document with an event in the month."""
    start = datetime.combine(month_start, datetime.min.time(), UTC)
    end = datetime.combine(month_end, datetime.min.time(), UTC)
    eic = str(REGIONS[region]["entsoe"])
    rows: list[dict[str, Any]] = []
    for query in entsoe_outage_queries(eic, ENTSOE_OUTAGE_BORDERS[region], start, end):
        rows.extend(fetch_outage_query(query, start, end, api_key, fetch))
    return rows


def load_entsoe_outages(
    region: str,
    first: date,
    last: date,
    cache_dir: Path,
    api_key: str,
    fetch: Callable[[str], bytes],
    today: date | None = None,
) -> OutageIndex:
    """Load the zone's ENTSO-E outages for events on the UTC days ``first``..``last``.

    One request series per calendar month (a long outage appears in each
    month it touches; the index counts a period once). Months that ended
    before ``today`` are cached as parsed rows in ``cache_dir``.
    """
    today = today or date.today()
    rows: list[dict[str, Any]] = []
    for month_start, month_end in months(first, last):
        cache_file = cache_dir / f"entsoe_outages_{region}_{month_start:%Y-%m}.json"
        if cache_file.exists():
            month = json.loads(cache_file.read_text(encoding="utf-8"))
        else:
            print(
                f"  ENTSO-E outages {region} {month_start:%Y-%m} ...", file=sys.stderr
            )
            month = fetch_entsoe_outage_month(
                region, month_start, month_end, api_key, fetch
            )
            if month_end < today:
                cache_dir.mkdir(parents=True, exist_ok=True)
                cache_file.write_text(json.dumps(month), encoding="utf-8")
        rows.extend(month)
    return OutageIndex(rows)
