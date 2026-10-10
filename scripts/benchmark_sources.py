"""Sources and storage of the external-forecast benchmark (dev-only, #115).

``scripts/benchmark.py collect`` snapshots every source's price forecast as
it is published and appends it to a SQLite file in the git-ignored
``.cache/benchmark/``: one row per ``(source, origin, slot, price)``, the
price in EUR/MWh, times in UTC unix seconds. ``origin`` is when the forecast
was published (or fetched), which decides what the report may score it for.

Sources and their terms (checked 2026-10-10):

- ``smartere``: Smartere Elforbrug (elforbrug.nu, EWII), ``prognose.json`` in
  the public repository solmoller/Spotprisprognose: hourly, EUR/MWh, ``Time``
  in UTC, ``Forecast created`` in Danish local time. Free for non-commercial
  consumer use; **no redistribution or refinement**: the forecasts stay in
  ``.cache/`` and only aggregate error numbers are published. Its git history
  holds every past forecast, so ``--backfill-days`` reconstructs them.
- ``carnot``: carnot.dk, a personal key (``CARNOT_API_KEY``, ``CARNOT_USER``)
  through the endpoint the Energi Data Service integration uses: hourly,
  DKK/MWh, ``utctime`` in UTC. Undocumented and personal.
- ``epex``: EpexPredictor's public instance (b3nn0, fair use, one request per
  run): 15-minute prices, EUR/MWh.
- ``osf``: Open Spot Forecast itself, as published at the same moment: the
  ``get_forecast`` action of a running instance (``HA_URL``, ``HA_TOKEN``;
  raw spot price, no tariffs, surcharge or VAT) or the pending predictions of
  a learning-database export, per 15 minutes.

Credentials are read from the environment only and sent as request headers;
they are never stored, printed or put in a URL. Not shipped with the
integration.
"""

import json
import os
import sqlite3
import urllib.error
import urllib.parse
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from custom_components.open_spot_forecast.api.exchange_rates import PEGGED_EUR_RATES

from .backtest import _http_get_json

SOURCES = ("smartere", "carnot", "epex", "osf")
BENCHMARK_REGIONS = ("DK1", "DK2")
# Smartere Elforbrug's "Forecast created" is Danish wall-clock time
SMARTERE_TZ = ZoneInfo("Europe/Copenhagen")
SMARTERE_REPOSITORY = "solmoller/Spotprisprognose"
SMARTERE_FILE = "prognose.json"
SMARTERE_RAW = "https://raw.githubusercontent.com/{repository}/{ref}/{file}"
GITHUB_COMMITS_API = "https://api.github.com/repos/{repository}/commits"
GITHUB_PAGE_SIZE = 100
CARNOT_API = "https://whale-app-dquqw.ondigitalocean.app/openapi/get_predict"
CARNOT_DAYS_AHEAD = 7
EPEX_API = "https://epexpredictor.batzill.com/prices"
EPEX_HOURS = 168
HA_GET_FORECAST = "/api/services/open_spot_forecast/get_forecast?return_response"
KWH_PER_MWH = 1000.0
# Predictions of one forecast run are stored within seconds of each other
OSF_RUN_GAP_SECONDS = 300

# Fetches a JSON document: (url, headers, request body) -> parsed JSON
FetchJson = Callable[[str, Mapping[str, str] | None, bytes | None], Any]

_SCHEMA = """
    CREATE TABLE IF NOT EXISTS forecasts (
        source          TEXT    NOT NULL,
        origin_utc      INTEGER NOT NULL,
        slot_start_utc  INTEGER NOT NULL,
        price_eur_mwh   REAL    NOT NULL,
        PRIMARY KEY (source, origin_utc, slot_start_utc)
    );
    CREATE TABLE IF NOT EXISTS backfilled (
        source  TEXT NOT NULL,
        ref     TEXT NOT NULL,
        PRIMARY KEY (source, ref)
    );
"""


class SourceUnavailable(Exception):
    """A source cannot be collected now (no credentials, nothing published)."""


@dataclass(frozen=True)
class Forecast:
    """One source's forecast as published at ``origin``.

    ``points`` are ``(slot start, EUR/MWh)`` in UTC unix seconds, at the
    source's own resolution (hourly or 15 minutes).
    """

    source: str
    origin: int
    points: tuple[tuple[int, float], ...]


def fetch_json(
    url: str, headers: Mapping[str, str] | None = None, data: bytes | None = None
) -> Any:
    """Fetch a JSON document with the backtest's retries (``_http_get_json``)."""
    return _http_get_json(url, headers=headers, data=data)


def _utc_seconds(value: str, naive_tz: ZoneInfo | None = None) -> int:
    """Return an ISO timestamp as UTC unix seconds; a naive one is UTC or ``naive_tz``."""
    moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=naive_tz or UTC)
    return int(moment.timestamp())


def eur_per_unit(currency: str) -> float:
    """Return how many EUR one unit of ``currency`` is (pegged rate for DKK).

    Raises:
        ValueError: The currency has no fixed rate here.
    """
    if currency == "EUR":
        return 1.0
    rate = PEGGED_EUR_RATES.get(currency)
    if rate is None:
        raise ValueError(f"no fixed EUR rate for {currency}")
    return 1.0 / rate


# --- Parsers (pure) ---------------------------------------------------------------


def parse_smartere(
    payload: Mapping[str, Any], region: str, committed: int | None = None
) -> Forecast:
    """Parse Smartere Elforbrug's ``prognose.json`` for one region.

    Args:
        payload: ``{"Forecast created": local time, "DK1": [{"Time": UTC,
            "Price": EUR/MWh}], "DK2": [...]}``.
        region: ``DK1`` or ``DK2``.
        committed: The git commit time, the origin when the file has no
            readable ``Forecast created``.

    Raises:
        ValueError: The file has neither an origin nor the region's prices.
    """
    created = payload.get("Forecast created")
    try:
        origin = _utc_seconds(str(created), SMARTERE_TZ)
    except ValueError:
        if committed is None:
            raise ValueError("prognose.json has no 'Forecast created'") from None
        origin = committed
    rows = payload.get(region)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"prognose.json has no {region} prices")
    return Forecast(
        "smartere",
        origin,
        tuple((_utc_seconds(str(row["Time"])), float(row["Price"])) for row in rows),
    )


def parse_carnot(payload: Mapping[str, Any], origin: int) -> Forecast:
    """Parse Carnot's ``get_predict`` response (DKK/MWh, ``utctime`` in UTC).

    Raises:
        ValueError: The response has no predictions.
    """
    rows = payload.get("predictions")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Carnot returned no predictions")
    rate = eur_per_unit("DKK")
    return Forecast(
        "carnot",
        origin,
        tuple(
            (_utc_seconds(str(row["utctime"])), float(row["prediction"]) * rate)
            for row in rows
        ),
    )


def parse_epex(payload: Mapping[str, Any], origin: int) -> Forecast:
    """Parse EpexPredictor's ``/prices`` response requested in ``EUR_PER_MWH``.

    Raises:
        ValueError: The response has no prices.
    """
    rows = payload.get("prices")
    if not isinstance(rows, list) or not rows:
        raise ValueError("EpexPredictor returned no prices")
    return Forecast(
        "epex",
        origin,
        tuple(
            (_utc_seconds(str(row["startsAt"])), float(row["total"])) for row in rows
        ),
    )


def parse_osf(payload: Mapping[str, Any], origin: int) -> Forecast:
    """Parse the ``get_forecast`` action's response (``raw``: spot price per kWh).

    The currency is the unit's (``DKK/kWh``), converted with the fixed rate.

    Raises:
        ValueError: The response has no forecast, or its unit is not per kWh.
    """
    response = payload.get("service_response", payload)
    rows = response.get("forecast")
    currency, _, per = str(response.get("unit", "")).partition("/")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Open Spot Forecast returned no forecast")
    if per != "kWh":
        raise ValueError(f"expected a price per kWh, got {response.get('unit')!r}")
    factor = eur_per_unit(currency) * KWH_PER_MWH
    return Forecast(
        "osf",
        origin,
        tuple(
            (_utc_seconds(str(row["start"])), float(row["price"]) * factor)
            for row in rows
        ),
    )


# --- Fetching -------------------------------------------------------------------


def _github_headers(environ: Mapping[str, str]) -> dict[str, str]:
    """Return the GitHub API headers, with ``GITHUB_TOKEN`` if it is set."""
    headers = {"Accept": "application/vnd.github+json"}
    if token := environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"
    return headers


def collect_smartere(region: str, fetch: FetchJson = fetch_json) -> Forecast:
    """Fetch the current Smartere Elforbrug forecast."""
    url = SMARTERE_RAW.format(
        repository=SMARTERE_REPOSITORY, ref="main", file=SMARTERE_FILE
    )
    return parse_smartere(fetch(url, None, None), region)


def smartere_commits(
    since: datetime,
    fetch: FetchJson = fetch_json,
    environ: Mapping[str, str] = os.environ,
) -> Iterator[tuple[str, int]]:
    """Yield ``(sha, commit time)`` of every ``prognose.json`` commit since ``since``.

    Newest first, one GitHub API page of ``GITHUB_PAGE_SIZE`` at a time
    (60 requests per hour without ``GITHUB_TOKEN``).
    """
    page = 1
    while True:
        query = urllib.parse.urlencode(
            {
                "path": SMARTERE_FILE,
                "since": since.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "per_page": GITHUB_PAGE_SIZE,
                "page": page,
            }
        )
        url = GITHUB_COMMITS_API.format(repository=SMARTERE_REPOSITORY)
        commits = fetch(f"{url}?{query}", _github_headers(environ), None)
        if not isinstance(commits, list):
            raise TypeError("unexpected answer from the GitHub commits API")
        for commit in commits:
            yield (
                str(commit["sha"]),
                _utc_seconds(str(commit["commit"]["committer"]["date"])),
            )
        if len(commits) < GITHUB_PAGE_SIZE:
            return
        page += 1


def backfill_smartere(
    store: BenchmarkStore,
    region: str,
    since: datetime,
    fetch: FetchJson = fetch_json,
    environ: Mapping[str, str] = os.environ,
) -> tuple[int, str | None]:
    """Store the past Smartere Elforbrug forecasts from its git history.

    Commits already stored are skipped, so a run cut short by GitHub's rate
    limit continues where it stopped when it is run again.

    Returns:
        The number of forecasts stored, and why the backfill stopped early
        (None if it finished).
    """
    stored = 0
    try:
        for sha, committed in smartere_commits(since, fetch, environ):
            if store.is_backfilled("smartere", sha):
                continue
            url = SMARTERE_RAW.format(
                repository=SMARTERE_REPOSITORY, ref=sha, file=SMARTERE_FILE
            )
            try:
                forecast = parse_smartere(fetch(url, None, None), region, committed)
            except ValueError, KeyError, TypeError:
                # A commit with an unreadable file: nothing to score
                store.mark_backfilled("smartere", sha)
                continue
            store.add(forecast)
            store.mark_backfilled("smartere", sha)
            stored += 1
    except urllib.error.HTTPError as err:
        if err.code not in (403, 429):
            raise
        return stored, (
            f"GitHub rate limit (HTTP {err.code}); run again later, or set GITHUB_TOKEN"
        )
    return stored, None


def collect_carnot(
    region: str,
    now: datetime,
    fetch: FetchJson = fetch_json,
    environ: Mapping[str, str] = os.environ,
) -> Forecast:
    """Fetch Carnot's forecast with the personal key from the environment.

    Raises:
        SourceUnavailable: ``CARNOT_API_KEY`` or ``CARNOT_USER`` is not set.
    """
    key, user = environ.get("CARNOT_API_KEY"), environ.get("CARNOT_USER")
    if not key or not user:
        raise SourceUnavailable("CARNOT_API_KEY and CARNOT_USER are not set")
    query = urllib.parse.urlencode(
        {
            "region": region.lower(),
            "energysource": "spotprice",
            "daysahead": CARNOT_DAYS_AHEAD,
        }
    )
    payload = fetch(f"{CARNOT_API}?{query}", {"apikey": key, "username": user}, None)
    return parse_carnot(payload, int(now.timestamp()))


def collect_epex(region: str, now: datetime, fetch: FetchJson = fetch_json) -> Forecast:
    """Fetch EpexPredictor's forecast (one request, 15-minute prices)."""
    query = urllib.parse.urlencode(
        {
            "region": region,
            "unit": "EUR_PER_MWH",
            "hours": EPEX_HOURS,
            "timezone": "UTC",
        }
    )
    return parse_epex(fetch(f"{EPEX_API}?{query}", None, None), int(now.timestamp()))


def collect_osf(
    now: datetime,
    fetch: FetchJson = fetch_json,
    environ: Mapping[str, str] = os.environ,
) -> Forecast:
    """Fetch the running instance's forecast through Home Assistant's REST API.

    The ``get_forecast`` action with ``raw`` returns the raw spot price per
    15 minutes, without the confirmed prices. ``OSF_CONFIG_ENTRY_ID`` picks
    the entry when the instance has several.

    Raises:
        SourceUnavailable: ``HA_URL`` or ``HA_TOKEN`` is not set.
    """
    base, token = environ.get("HA_URL"), environ.get("HA_TOKEN")
    if not base or not token:
        raise SourceUnavailable("HA_URL and HA_TOKEN are not set")
    body: dict[str, Any] = {"raw": True, "hourly": False, "include_known": False}
    if entry_id := environ.get("OSF_CONFIG_ENTRY_ID"):
        body["config_entry_id"] = entry_id
    payload = fetch(
        base.rstrip("/") + HA_GET_FORECAST,
        {"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json.dumps(body).encode(),
    )
    return parse_osf(payload, int(now.timestamp()))


def load_osf_export(path: Path, currency: str, tz: ZoneInfo) -> list[Forecast]:
    """Read the pending predictions of a learning-database export, per forecast run.

    The ``predictions`` table holds every stored prediction of the slots not
    yet scored, with the time it was stored. Rows stored within
    ``OSF_RUN_GAP_SECONDS`` of each other are one run, whose origin is its
    first row's time. The export is opened read-only.

    Args:
        path: The export (see ``docs/persistence.md``).
        currency: Currency of the instance's prices (per kWh).
        tz: The instance's time zone, for timestamps stored without an offset.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    if not path.is_file():
        raise FileNotFoundError(path)
    with closing(
        sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    ) as connection:
        rows = connection.execute(
            "SELECT start, price, stored_at FROM predictions"
        ).fetchall()
    factor = eur_per_unit(currency) * KWH_PER_MWH
    stored = sorted(
        (_utc_seconds(str(at), tz), _utc_seconds(str(start), tz), float(price) * factor)
        for start, price, at in rows
    )
    runs: list[list[tuple[int, int, float]]] = []
    for row in stored:
        if not runs or row[0] - runs[-1][-1][0] > OSF_RUN_GAP_SECONDS:
            runs.append([])
        runs[-1].append(row)
    return [
        Forecast("osf", run[0][0], tuple(sorted({s: p for _, s, p in run}.items())))
        for run in runs
    ]


# --- Storage --------------------------------------------------------------------


class BenchmarkStore:
    """The collected forecasts of one region, in a SQLite file."""

    def __init__(self, path: Path) -> None:
        """Open (and create) the store at ``path``."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._connection = sqlite3.connect(path)
        self._connection.executescript(_SCHEMA)

    def close(self) -> None:
        """Close the store."""
        self._connection.close()

    def add(self, forecast: Forecast) -> int:
        """Store a forecast; rows already stored are left alone (idempotent).

        Returns:
            The number of new rows.
        """
        with self._connection:
            cursor = self._connection.executemany(
                "INSERT OR IGNORE INTO forecasts VALUES (?, ?, ?, ?)",
                [
                    (forecast.source, forecast.origin, start, price)
                    for start, price in forecast.points
                ],
            )
        return cursor.rowcount

    def unchanged_since(self, forecast: Forecast) -> int | None:
        """Return the origin of the source's latest forecast if it has the same prices.

        A source fetched now has no publication time of its own; fetched again
        before it changes, it is the forecast already stored, published no
        later than that one's origin.
        """
        row = self._connection.execute(
            "SELECT MAX(origin_utc) FROM forecasts WHERE source = ?",
            (forecast.source,),
        ).fetchone()
        if row[0] is None:
            return None
        latest = dict(
            self._connection.execute(
                "SELECT slot_start_utc, price_eur_mwh FROM forecasts"
                " WHERE source = ? AND origin_utc = ?",
                (forecast.source, row[0]),
            )
        )
        points = dict(forecast.points)
        same = latest.keys() == points.keys() and all(
            abs(latest[start] - price) < 1e-9 for start, price in points.items()
        )
        return int(row[0]) if same else None

    def is_backfilled(self, source: str, ref: str) -> bool:
        """Return whether a historical forecast (a commit) was already stored."""
        row = self._connection.execute(
            "SELECT 1 FROM backfilled WHERE source = ? AND ref = ?", (source, ref)
        ).fetchone()
        return row is not None

    def mark_backfilled(self, source: str, ref: str) -> None:
        """Remember that a historical forecast was stored."""
        with self._connection:
            self._connection.execute(
                "INSERT OR IGNORE INTO backfilled VALUES (?, ?)", (source, ref)
            )

    def count(self, source: str | None = None) -> int:
        """Return the number of stored rows, of one source or of all."""
        if source is None:
            row = self._connection.execute("SELECT COUNT(*) FROM forecasts")
        else:
            row = self._connection.execute(
                "SELECT COUNT(*) FROM forecasts WHERE source = ?", (source,)
            )
        return int(row.fetchone()[0])

    def forecasts(
        self, sources: Iterable[str]
    ) -> dict[str, dict[int, dict[int, float]]]:
        """Return ``{source: {origin: {slot start: EUR/MWh}}}`` of the sources."""
        wanted: Sequence[str] = list(sources)
        loaded: dict[str, dict[int, dict[int, float]]] = {}
        marks = ", ".join("?" * len(wanted))
        for source, origin, start, price in self._connection.execute(
            "SELECT source, origin_utc, slot_start_utc, price_eur_mwh FROM forecasts"
            f" WHERE source IN ({marks}) ORDER BY origin_utc, slot_start_utc",
            wanted,
        ):
            loaded.setdefault(source, {}).setdefault(origin, {})[start] = price
        return loaded


def default_store_path(region: str, cache_dir: Path) -> Path:
    """Return the region's store in the cache directory."""
    return cache_dir / "benchmark" / f"{region}.db"


def origin_time(origin: int) -> str:
    """Return an origin as a UTC timestamp for messages."""
    return (datetime(1970, 1, 1, tzinfo=UTC) + timedelta(seconds=origin)).strftime(
        "%Y-%m-%d %H:%MZ"
    )
