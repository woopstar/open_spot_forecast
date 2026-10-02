"""Rolling backtest of the Open Spot Forecast price model (dev-only).

Gives every ML change a reproducible before/after number for multi-day accuracy.
The method reimplements EpexPredictor's ``predictor/performance_testing.py``
(BSD-3-Clause, https://github.com/b3nn0/EpexPredictor): for every day of a test
period the models are retrained on the preceding ``--window-days`` of prices,
forecast the next three local days, and each day is scored separately, giving
MAE and RMSE for 1, 2 and 3 days ahead.

Leakage is prevented by a strict horizon cutoff, after EpexPredictor's
``DataStore.horizon_cutoff``: a model only ever receives the read-only copy
returned by ``PriceSeries.between(window_start, cutoff)``, so neither training
nor feature building can see a price at or after the forecast origin.

The "current" model is built from the integration's own ``build_feature_row``,
``build_feature_vector`` and ``create_price_model``; no feature or model logic
is duplicated here. This script is not shipped with the integration. Run it
from the repository root::

    python -m scripts.backtest --region DK1
    ./scripts/quality.sh backtest --region DK1 --window-days 30

The optional LightGBM reference row needs
``pip install -r requirements_backtest.txt``.
"""

import argparse
import importlib.util
import json
import logging
import math
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta, tzinfo
from pathlib import Path
from typing import Any, Protocol
from zoneinfo import ZoneInfo

import numpy as np

from custom_components.open_spot_forecast.api.dayahead_prices import (
    parse_energy_charts as parse_dayahead_rows,
)
from custom_components.open_spot_forecast.api.entsoe import entsoe_period
from custom_components.open_spot_forecast.api.entsoe_load import (
    load_curve,
    parse_entsoe_load,
)
from custom_components.open_spot_forecast.api.gas_prices import (
    instrat_query,
    parse_instrat_gas,
)
from custom_components.open_spot_forecast.api.openmeteo_weather import (
    open_meteo_query,
    parse_open_meteo,
)
from custom_components.open_spot_forecast.const import (
    ENERGY_CHARTS_API,
    ENTSOE_API,
    INSTRAT_GAS_API,
    NEIGHBOURS,
    OPEN_METEO_ARCHIVE_API,
    REGIONS,
    WEATHER_POINTS,
)
from custom_components.open_spot_forecast.ml.cross_border import (
    Stage1Model,
    cross_price_name,
    local_day_fold,
    neighbour_feature_rows,
)
from custom_components.open_spot_forecast.ml.features import (
    FEATURE_NAMES,
    SlotInputs,
    build_feature_row,
    build_feature_vector,
    with_masked_nordpool,
)
from custom_components.open_spot_forecast.ml.gas_price import (
    GAS_LOOKBACK_DAYS,
    GasPriceIndex,
)
from custom_components.open_spot_forecast.ml.gbm import NumpyGradientBoosting
from custom_components.open_spot_forecast.ml.models import create_price_model
from custom_components.open_spot_forecast.ml.public_holidays import public_holiday
from custom_components.open_spot_forecast.ml.zone_weather import ZoneWeatherIndex
from custom_components.open_spot_forecast.time_series import iso_weeks

from .backtest_lags import LagConfig, PriceLagIndex
from .backtest_nordpool import NordpoolInputs, load_nordpool_db

SLOT_SECONDS = 15 * 60
# Origins with less history than this are skipped (the naive model needs a week).
MIN_HISTORY_SLOTS = 7 * 96

HTTP_ATTEMPTS = 5
# First wait before retrying a transient failure; doubles per attempt.
TRANSIENT_BACKOFF_SECONDS = 2.0
# OSF regions and their energy-charts bidding zones (``REGIONS``, #27)
ENERGY_CHARTS_ZONES: dict[str, str] = {
    region: str(zone["energy_charts"]) for region, zone in REGIONS.items()
}
OPEN_METEO_REQUEST_DAYS = 90
WEATHER_CHOICES = ("openmeteo", "none")

# Report in EUR ct/kWh, the unit EpexPredictor publishes, so numbers compare directly.
CT_PER_KWH_PER_EUR_PER_MWH = 0.1

MODEL_NAMES = ("naive", "current", "lightgbm")

# Fixed LightGBM reference configuration (see docs/ml_documentation.md).
LIGHTGBM_PARAMS: dict[str, Any] = {
    "objective": "regression",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 20,
    "seed": 42,
    "deterministic": True,
    "force_row_wise": True,
    # One thread: on ~6k rows OpenMP's per-core threads mostly contend, and
    # on a loaded machine a fit slowed from under a second to half a minute
    "num_threads": 1,
    "verbosity": -1,
}
LIGHTGBM_ROUNDS = 500


def _read_only(values: np.ndarray) -> np.ndarray:
    """Return a read-only copy, so the slice keeps no link to the full array."""
    copy = values.copy()
    copy.flags.writeable = False
    return copy


@dataclass(frozen=True)
class PriceSeries:
    """Prices on a 15-minute grid, keyed by slot start in UTC unix seconds."""

    starts: np.ndarray
    prices: np.ndarray

    def __post_init__(self) -> None:
        """Reject misaligned or unsorted input."""
        if self.starts.ndim != 1 or self.starts.shape != self.prices.shape:
            raise ValueError("starts and prices must be 1-D arrays of equal length")
        if self.starts.size > 1 and not bool(np.all(np.diff(self.starts) > 0)):
            raise ValueError("starts must be strictly increasing")

    @classmethod
    def from_points(cls, starts: Sequence[int], prices: Sequence[float]) -> PriceSeries:
        """Build a series from unordered points, keeping the first of any duplicate."""
        start_array = np.asarray(starts, dtype=np.int64)
        price_array = np.asarray(prices, dtype=float)
        unique, first_index = np.unique(start_array, return_index=True)
        return cls(unique, price_array[first_index])

    def __len__(self) -> int:
        """Return the number of slots."""
        return int(self.starts.size)

    def between(self, start: int, cutoff: int) -> PriceSeries:
        """Return the slots with ``start <= slot start < cutoff``.

        ``cutoff`` is the horizon cutoff: the slot beginning exactly at it is
        excluded. The arrays are read-only copies rather than views, so a model
        cannot reach later prices through ``ndarray.base``.
        """
        low = int(np.searchsorted(self.starts, start, side="left"))
        high = int(np.searchsorted(self.starts, cutoff, side="left"))
        return PriceSeries(
            _read_only(self.starts[low:high]), _read_only(self.prices[low:high])
        )

    def prices_at(self, starts: np.ndarray) -> np.ndarray:
        """Return the price for each requested slot start, NaN where none exists."""
        result = np.full(starts.shape, math.nan)
        if not len(self):
            return result
        index = np.searchsorted(self.starts, starts)
        clipped = np.minimum(index, len(self) - 1)
        found = (index < len(self)) & (self.starts[clipped] == starts)
        result[found] = self.prices[clipped[found]]
        return result


def merge_series(parts: Sequence[PriceSeries]) -> PriceSeries:
    """Concatenate series, keeping the first price seen for a duplicated slot."""
    if not parts:
        return PriceSeries.from_points([], [])
    return PriceSeries.from_points(
        np.concatenate([part.starts for part in parts]).tolist(),
        np.concatenate([part.prices for part in parts]).tolist(),
    )


# --- Data: energy-charts day-ahead prices -------------------------------------


def parse_energy_charts(payload: dict[str, Any]) -> PriceSeries:
    """Convert an energy-charts ``/price`` response to a 15-minute ct/kWh series.

    Parsed by the integration's ``parse_energy_charts`` (``api/dayahead_prices.py``):
    each price covers the interval up to the next timestamp, capped at one
    hour, so hourly prices (DK1 before the 2025-10-01 move to 15-minute
    products) fill four quarter-hours, and a null price never stretches a
    neighbouring one.
    """
    rows = parse_dayahead_rows(payload)
    return PriceSeries.from_points(
        [int(datetime.fromisoformat(row["timestamp"]).timestamp()) for row in rows],
        [float(row["price"]) * CT_PER_KWH_PER_EUR_PER_MWH for row in rows],
    )


def _month_chunks(first: date, last: date) -> Iterator[tuple[date, date]]:
    """Yield (first day, last day) of every calendar month touching [first, last]."""
    current = first.replace(day=1)
    while current <= last:
        following = (current.replace(day=28) + timedelta(days=4)).replace(day=1)
        yield current, following - timedelta(days=1)
        current = following


def _retry_after_seconds(value: str | None) -> float:
    """Parse a Retry-After header (seconds form), bounded to 1-300 s."""
    try:
        return min(max(float(value or ""), 1.0), 300.0)
    except ValueError:
        return 30.0


def _http_get_json(
    url: str, sleep: Callable[[float], None] = time.sleep
) -> dict[str, Any]:
    """Fetch and decode a JSON document, retrying transient failures.

    HTTP 429 waits out the server's ``Retry-After``; a truncated or invalid
    body, a connection error or timeout and an HTTP 5xx are retried with a
    doubling back-off (``TRANSIENT_BACKOFF_SECONDS`` first). Other HTTP
    errors (404, 400, ...) are raised at once. At most ``HTTP_ATTEMPTS``
    attempts, then the last error propagates.
    """
    request = urllib.request.Request(
        url, headers={"User-Agent": "open-spot-forecast-backtest"}
    )
    attempt = 1
    while True:
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload: dict[str, Any] = json.load(response)
                return payload
        except urllib.error.HTTPError as err:
            if err.code == 429:
                reason = "rate limited"
                wait = _retry_after_seconds(err.headers.get("Retry-After"))
            elif err.code >= 500:
                reason = f"HTTP {err.code}"
                wait = TRANSIENT_BACKOFF_SECONDS * 2 ** (attempt - 1)
            else:
                raise
            if attempt >= HTTP_ATTEMPTS:
                raise
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as err:
            if attempt >= HTTP_ATTEMPTS:
                raise
            reason = (
                "truncated response"
                if isinstance(err, json.JSONDecodeError)
                else str(err)
            )
            wait = TRANSIENT_BACKOFF_SECONDS * 2 ** (attempt - 1)
        print(f"  {reason}, retrying in {wait:.0f} s", file=sys.stderr)
        sleep(wait)
        attempt += 1


def _read_cache(cache_file: Path) -> Any | None:
    """Return a cached JSON document, or None when it is missing or unreadable.

    A file an interrupted run left truncated is deleted and fetched again
    instead of failing every later run.
    """
    if not cache_file.exists():
        return None
    try:
        return json.loads(cache_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        print(f"  discarding unreadable cache file {cache_file.name}", file=sys.stderr)
        cache_file.unlink()
        return None


def _write_cache(cache_file: Path, document: Any) -> None:
    """Write a JSON document atomically: a temp file next to it, then a rename.

    The final name only appears once the content is complete, so a run killed
    while writing (Ctrl-C, a time limit) cannot leave a half-written file.
    """
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    temp_file = cache_file.with_name(cache_file.name + ".tmp")
    temp_file.write_text(json.dumps(document), encoding="utf-8")
    os.replace(temp_file, cache_file)


def load_energy_charts_prices(
    region: str,
    first: date,
    last: date,
    cache_dir: Path,
    fetch: Callable[[str], dict[str, Any]] = _http_get_json,
    today: date | None = None,
) -> PriceSeries:
    """Load day-ahead prices for the local dates ``first``..``last``.

    Prices are fetched one calendar month per request. Complete past months
    are cached as JSON in ``cache_dir``, so repeated runs work offline.
    energy-charts answers a month without any price yet (e.g. a horizon
    reaching into next month) with HTTP 404; such a month is empty, not cached.
    Prices are © Bundesnetzagentur | SMARD.de, CC BY 4.0, via energy-charts.info.
    """
    zone = ENERGY_CHARTS_ZONES[region]
    today = today or date.today()
    parts = []
    for month_start, month_end in _month_chunks(first, last):
        cache_file = cache_dir / f"energy_charts_{zone}_{month_start:%Y-%m}.json"
        payload = _read_cache(cache_file)
        if payload is None:
            query = urllib.parse.urlencode(
                {
                    "bzn": zone,
                    "start": month_start.isoformat(),
                    "end": month_end.isoformat(),
                }
            )
            try:
                payload = fetch(f"{ENERGY_CHARTS_API}?{query}")
            except urllib.error.HTTPError as err:
                if err.code != 404:
                    raise
                print(f"  no {zone} prices for {month_start:%Y-%m}", file=sys.stderr)
                continue
            if month_end < today:
                _write_cache(cache_file, payload)
        parts.append(parse_energy_charts(payload))
    return merge_series(parts)


def _http_get_text(url: str) -> str:
    """Fetch a text document; an HTTP 400 answer is returned too.

    ENTSO-E answers "no matching data" with 400 and an acknowledgement
    document. The URL (with the ENTSO-E token) is never printed.
    """
    request = urllib.request.Request(
        url, headers={"User-Agent": "open-spot-forecast-backtest"}
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return str(response.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        if err.code != 400:
            raise
        return str(err.read().decode("utf-8"))


def load_entsoe_load(
    region: str,
    first: date,
    last: date,
    cache_dir: Path,
    api_key: str,
    fetch: Callable[[str], str] = _http_get_text,
    today: date | None = None,
) -> dict[int, float]:
    """Load ENTSO-E's week-ahead load forecast for the local days ``first``..``last``.

    ENTSO-E answers one week-ahead document per request, the ISO week of
    ``periodStart``, so there is one request per ISO week (A65/A31). The
    daily (minimum, maximum) values of complete past weeks are cached in
    ``cache_dir``; the token is never cached or printed. The days become the
    integration's 15-minute curve (``load_curve``), keyed by slot start (UTC
    epoch). A week ENTSO-E has no data for (an acknowledgement) has no days.
    """
    zone = REGIONS[region]
    tz = ZoneInfo(str(zone["tz"]))
    today = today or date.today()
    days: dict[date, tuple[float, float]] = {}
    for monday in iso_weeks(
        datetime.combine(first, datetime.min.time(), tz),
        datetime.combine(last + timedelta(days=1), datetime.min.time(), tz),
        tz,
    ):
        cache_file = cache_dir / f"entsoe_load_{region}_{monday:%G-W%V}.json"
        cached = _read_cache(cache_file)
        if cached is not None:
            week = {date.fromisoformat(k): (v[0], v[1]) for k, v in cached.items()}
        else:
            week_start = datetime.combine(monday, datetime.min.time(), tz)
            query = urllib.parse.urlencode(
                {
                    "documentType": "A65",
                    "processType": "A31",
                    "outBiddingZone_Domain": str(zone["entsoe"]),
                    **entsoe_period(week_start, week_start + timedelta(weeks=1)),
                    "securityToken": api_key,
                }
            )
            week = parse_entsoe_load(fetch(f"{ENTSOE_API}?{query}"), tz)
            if monday + timedelta(weeks=1) <= today:
                _write_cache(
                    cache_file, {k.isoformat(): list(v) for k, v in week.items()}
                )
        days.update(week)
    return {
        int(datetime.fromisoformat(row["timestamp"]).timestamp()): row["load"]
        for row in load_curve(days, tz)
    }


def load_open_meteo_weather(
    region: str,
    first: date,
    last: date,
    cache_dir: Path,
    fetch: Callable[[str], Any] = _http_get_json,
    today: date | None = None,
) -> ZoneWeatherIndex:
    """Load the region's zone weather for the UTC days ``first``..``last``.

    Open-Meteo's archived forecasts (``historical-forecast-api``), 90 days
    per request, for the region's ``WEATHER_POINTS``. Complete past chunks
    are cached as JSON in ``cache_dir``. Weather data by Open-Meteo.com,
    CC BY 4.0.
    """
    points = WEATHER_POINTS[region]
    today = today or date.today()
    rows: list[dict[str, Any]] = []
    chunk_start = first
    while chunk_start <= last:
        chunk_end = min(chunk_start + timedelta(days=OPEN_METEO_REQUEST_DAYS - 1), last)
        cache_file = cache_dir / f"openmeteo_{region}_{chunk_start}_{chunk_end}.json"
        payload = _read_cache(cache_file)
        if payload is None:
            query = urllib.parse.urlencode(
                open_meteo_query(points, chunk_start, chunk_end)
            )
            payload = fetch(f"{OPEN_METEO_ARCHIVE_API}?{query}")
            if chunk_end < today - timedelta(days=2):
                _write_cache(cache_file, payload)
        rows.extend(parse_open_meteo(payload, points))
        chunk_start = chunk_end + timedelta(days=1)
    return ZoneWeatherIndex(rows)


def load_gas_prices(
    first: date,
    last: date,
    cache_dir: Path,
    fetch: Callable[[str], Any] = _http_get_json,
    today: date | None = None,
) -> list[dict[str, Any]]:
    """Load the daily gas prices (#28) for the UTC days ``first``..``last``.

    Instrat's TGE gas day-ahead index, one request per calendar month;
    complete past months are cached in ``cache_dir``. CC BY-NC 4.0,
    energy.instrat.pl.
    """
    today = today or date.today()
    rows: list[dict[str, Any]] = []
    for month_start, month_end in _month_chunks(first, last):
        cache_file = cache_dir / f"gas_instrat_{month_start:%Y-%m}.json"
        payload = _read_cache(cache_file)
        if payload is None:
            start = datetime.combine(month_start, datetime.min.time(), UTC)
            end = datetime.combine(
                month_end + timedelta(days=1), datetime.min.time(), UTC
            )
            query = urllib.parse.urlencode(instrat_query(start, end))
            payload = fetch(f"{INSTRAT_GAS_API}?{query}")
            if month_end < today - timedelta(days=2):
                _write_cache(cache_file, payload)
        rows.extend(parse_instrat_gas(payload))
    return rows


def load_cross_border(
    region: str,
    first: date,
    last: date,
    cache_dir: Path,
    with_weather: bool = True,
) -> CrossBorderInputs:
    """Load the neighbouring zones' prices and zone weather (#29)."""
    prices: dict[str, PriceSeries] = {}
    weather: dict[str, ZoneWeatherIndex | None] = {}
    for zone in NEIGHBOURS[region]:
        print(f"Loading neighbour {zone} ...", file=sys.stderr)
        prices[zone] = load_energy_charts_prices(zone, first, last, cache_dir)
        weather[zone] = (
            load_open_meteo_weather(
                zone, first - timedelta(days=1), last + timedelta(days=1), cache_dir
            )
            if with_weather
            else None
        )
    return CrossBorderInputs(ZoneInfo(str(REGIONS[region]["tz"])), prices, weather)


# --- Models --------------------------------------------------------------------


def feature_matrix(
    starts: np.ndarray,
    tz: tzinfo,
    zone: ZoneWeatherIndex | None = None,
    region: str | None = None,
    load: Mapping[int, float] | None = None,
    gas: GasPriceIndex | None = None,
    nordpool: NordpoolInputs | None = None,
    nordpool_until: int | None = None,
) -> np.ndarray:
    """Return the integration's model input for each slot start.

    Rows come from ``build_feature_row``, as in training and prediction. The
    zone weather (#22) comes from Open-Meteo's archived forecasts, for
    training and target slots alike, the sun features (#25) from the
    ``region``'s zone centre, ``load`` is ENTSO-E's week-ahead load
    forecast by slot start (#30) and ``gas`` the gas prices (#28). The local
    weather entity has no year of history, so it is unknown (NaN), and so
    are the Nordpool prognoses unless ``nordpool`` holds a live export's
    (#91): slots from ``nordpool_until`` on get none.
    """
    rows = []
    for start in starts.tolist():
        moment = datetime.fromtimestamp(start, tz)
        inputs = SlotInputs(
            **(zone.for_slot(moment) if zone else {}),
            **(nordpool.slot_inputs(start, nordpool_until) if nordpool else {}),
            load_forecast=load.get(start) if load else None,
            gas_price=gas.before(moment.date()) if gas else None,
        )
        rows.append(build_feature_vector(build_feature_row(moment, inputs, region)))
    return np.array(rows, dtype=float).reshape(len(rows), len(FEATURE_NAMES))


@dataclass
class CrossBorderInputs:
    """Neighbouring zones' prices and weather for the two-stage model (#29).

    Stage 1 is the integration's ``Stage1Model`` per neighbour, fitted on
    that zone's prices from the same window as the region's history and cut
    at the same horizon cutoff: training slots get the out-of-sample price,
    target slots the fold models' mean, as in ``CrossBorderModels``.
    """

    tz: tzinfo
    prices: Mapping[str, PriceSeries]
    weather: Mapping[str, ZoneWeatherIndex | None]
    _rows: dict[str, dict[int, np.ndarray]] = field(default_factory=dict, init=False)
    # The last origin's columns, shared by the NumPy and LightGBM rows
    _last: tuple[tuple[int, int], tuple[np.ndarray, np.ndarray]] | None = field(
        default=None, init=False
    )

    @property
    def names(self) -> list[str]:
        """Return the stage-2 column names, in neighbour order."""
        return [cross_price_name(zone) for zone in self.prices]

    def _feature_rows(self, zone: str, starts: np.ndarray) -> np.ndarray:
        """Return the zone's stage-1 rows for slot starts (cached: no prices)."""
        cache = self._rows.setdefault(zone, {})
        missing = [start for start in starts.tolist() if start not in cache]
        if missing:
            moments = [datetime.fromtimestamp(start, self.tz) for start in missing]
            rows = neighbour_feature_rows(zone, moments, self.weather.get(zone))
            cache.update(zip(missing, rows, strict=True))
        return np.array([cache[start] for start in starts.tolist()]).reshape(
            len(starts), len(FEATURE_NAMES)
        )

    def _folds(self, starts: np.ndarray) -> np.ndarray:
        return np.array(
            [
                local_day_fold(datetime.fromtimestamp(s, self.tz))
                for s in starts.tolist()
            ]
        )

    def columns(
        self, history: PriceSeries, targets: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the stage-1 columns of the training and the target slots.

        Only neighbour prices from the history's first slot up to the horizon
        cutoff (the first target slot) are used.
        """
        key = (int(history.starts.min()), int(targets.min()))
        if self._last is not None and self._last[0] == key:
            return self._last[1]
        train = np.full((len(history), len(self.prices)), math.nan)
        target = np.full((len(targets), len(self.prices)), math.nan)
        for index, (zone, series) in enumerate(self.prices.items()):
            visible = series.between(*key)
            model = Stage1Model()
            model.fit(
                self._feature_rows(zone, visible.starts),
                visible.prices,
                self._folds(visible.starts),
            )
            train[:, index] = model.predict_out_of_sample(
                self._feature_rows(zone, history.starts), self._folds(history.starts)
            )
            target[:, index] = model.predict(self._feature_rows(zone, targets))
        self._last = (key, (train, target))
        return train, target


def _feature_rows(
    history: PriceSeries,
    targets: np.ndarray,
    tz: tzinfo,
    zone: ZoneWeatherIndex | None = None,
    region: str | None = None,
    load: Mapping[int, float] | None = None,
    cross: CrossBorderInputs | None = None,
    gas: Sequence[dict[str, Any]] | None = None,
    nordpool: NordpoolInputs | None = None,
    lags: LagConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (training rows, target rows) for the slot times and inputs.

    With ``lags`` the lagged price columns (#119, ``PriceLagIndex`` on the
    visible history) and with ``cross`` the stage-1 price columns (#29)
    follow the features, in that order.
    ``gas`` holds the daily gas price rows (#28); only those dated before
    the horizon cutoff are used, so a target day gets the latest price
    published before the forecast, as in production. With ``nordpool``
    (#91) training rows get their stored prognoses and target rows none,
    or only the first forecast day with ``nordpool.day1``.
    """
    index = None
    first_target = datetime.fromtimestamp(int(targets.min()), tz)
    if gas is not None:
        cutoff = first_target.date()
        index = GasPriceIndex(
            row for row in gas if date.fromisoformat(row["timestamp"][:10]) < cutoff
        )
    until = int(targets.min())
    if nordpool is not None and nordpool.day1:
        until = local_midnight(first_target.date() + timedelta(days=1), tz)
    train = feature_matrix(history.starts, tz, zone, region, load, index, nordpool)
    target = feature_matrix(
        targets, tz, zone, region, load, index, nordpool, nordpool_until=until
    )
    if lags is not None:
        lag_index = PriceLagIndex(
            dict(zip(history.starts.tolist(), history.prices.tolist(), strict=True)),
            tz,
            lags.mixed_ages,
        )
        train = np.column_stack([train, lag_index.columns(history.starts, lags.names)])
        target = np.column_stack([target, lag_index.columns(targets, lags.names)])
    if cross is None:
        return train, target
    train_columns, target_columns = cross.columns(history, targets)
    return np.column_stack([train, train_columns]), np.column_stack(
        [target, target_columns]
    )


def _training_set(
    rows: np.ndarray, prices: np.ndarray, nordpool: NordpoolInputs | None
) -> tuple[np.ndarray, np.ndarray]:
    """Return the fitted rows: with prognoses, their masked copies (#91)."""
    if nordpool is None or not nordpool.copies:
        return rows, prices
    return with_masked_nordpool(rows, prices)


class Forecaster(Protocol):
    """A model under test. It only ever sees data before the horizon cutoff."""

    @property
    def name(self) -> str:
        """Row label in the report."""
        ...

    def forecast(self, history: PriceSeries, targets: np.ndarray) -> np.ndarray:
        """Return one price per target slot start, using only ``history``."""
        ...


@dataclass
class NaiveLastWeek:
    """Repeat the same local wall-clock slot from seven days earlier."""

    tz: tzinfo
    name: str = "naive (same slot last week)"

    def forecast(self, history: PriceSeries, targets: np.ndarray) -> np.ndarray:
        """Look up each target's slot one week back; fall back to the last price."""
        known = dict(zip(history.starts.tolist(), history.prices.tolist(), strict=True))
        fallback = float(history.prices[-1]) if len(history) else math.nan
        predicted = np.empty(targets.shape)
        for i, start in enumerate(targets.tolist()):
            week_ago = datetime.fromtimestamp(start, self.tz) - timedelta(days=7)
            predicted[i] = known.get(int(week_ago.timestamp()), fallback)
        return predicted


@dataclass
class CurrentModel:
    """OSF's production price model (NumPy GBM) on the integration's features."""

    tz: tzinfo
    model_factory: Callable[[], NumpyGradientBoosting] = create_price_model
    name: str = "current (NumPy GBM)"
    zone: ZoneWeatherIndex | None = None
    region: str | None = None
    load: Mapping[int, float] | None = None
    cross: CrossBorderInputs | None = None
    gas: Sequence[dict[str, Any]] | None = None
    nordpool: NordpoolInputs | None = None
    lags: LagConfig | None = None

    def forecast(self, history: PriceSeries, targets: np.ndarray) -> np.ndarray:
        """Fit a fresh model on ``history`` and predict the targets."""
        train_rows, target_rows = _feature_rows(
            history,
            targets,
            self.tz,
            self.zone,
            self.region,
            self.load,
            self.cross,
            self.gas,
            self.nordpool,
            self.lags,
        )
        model = self.model_factory()
        model.fit(*_training_set(train_rows, history.prices, self.nordpool))
        return model.predict(target_rows)


@dataclass
class LightGbmReference:
    """LightGBM on the same features, measuring the NumPy GBM's gap (#14).

    Dev-only: LightGBM has no musllinux wheels, so it cannot ship in the
    integration (Home Assistant OS and Container are Alpine-based).
    """

    tz: tzinfo
    name: str = "lightgbm (reference)"
    zone: ZoneWeatherIndex | None = None
    region: str | None = None
    load: Mapping[int, float] | None = None
    cross: CrossBorderInputs | None = None
    gas: Sequence[dict[str, Any]] | None = None
    nordpool: NordpoolInputs | None = None
    lags: LagConfig | None = None

    def forecast(self, history: PriceSeries, targets: np.ndarray) -> np.ndarray:
        """Fit a fresh LightGBM booster on ``history`` and predict the targets."""
        import lightgbm  # optional dev-only dependency (requirements_backtest.txt)

        train_rows, target_rows = _feature_rows(
            history,
            targets,
            self.tz,
            self.zone,
            self.region,
            self.load,
            self.cross,
            self.gas,
            self.nordpool,
            self.lags,
        )
        names = [
            *FEATURE_NAMES,
            *(self.lags.names if self.lags else ()),
            *(self.cross.names if self.cross else ()),
        ]
        rows, prices = _training_set(train_rows, history.prices, self.nordpool)
        dataset = lightgbm.Dataset(rows, label=prices, feature_name=names)
        booster = lightgbm.train(
            LIGHTGBM_PARAMS, dataset, num_boost_round=LIGHTGBM_ROUNDS
        )
        return np.asarray(booster.predict(target_rows), dtype=float)


def _lightgbm_available() -> bool:
    """Return True if the optional lightgbm package is importable."""
    return importlib.util.find_spec("lightgbm") is not None


def build_models(
    names: Sequence[str],
    tz: tzinfo,
    zone: ZoneWeatherIndex | None = None,
    region: str | None = None,
    load: Mapping[int, float] | None = None,
    cross: CrossBorderInputs | None = None,
    gas: Sequence[dict[str, Any]] | None = None,
    nordpool: NordpoolInputs | None = None,
    lags: LagConfig | None = None,
) -> tuple[list[Forecaster], dict[str, str]]:
    """Instantiate the requested models; return them plus {skipped name: reason}.

    With ``cross`` the GBM rows are two-stage cross-border models (#29);
    ``nordpool`` gives them a live export's prognoses (#91) and ``lags``
    the lagged price columns (#119).
    """
    models: list[Forecaster] = []
    skipped: dict[str, str] = {}
    stages = ", two-stage" if cross else ""
    for name in names:
        if name == "naive":
            models.append(NaiveLastWeek(tz))
        elif name == "current":
            models.append(
                CurrentModel(
                    tz,
                    name=f"current (NumPy GBM{stages})",
                    zone=zone,
                    region=region,
                    load=load,
                    cross=cross,
                    gas=gas,
                    nordpool=nordpool,
                    lags=lags,
                )
            )
        elif name == "lightgbm":
            if _lightgbm_available():
                models.append(
                    LightGbmReference(
                        tz,
                        name=f"lightgbm (reference{stages})",
                        zone=zone,
                        region=region,
                        load=load,
                        cross=cross,
                        gas=gas,
                        nordpool=nordpool,
                        lags=lags,
                    )
                )
            else:
                skipped[f"lightgbm (reference{stages})"] = (
                    "lightgbm not installed (pip install -r requirements_backtest.txt)"
                )
        else:
            raise ValueError(
                f"unknown model {name!r}; choose from {', '.join(MODEL_NAMES)}"
            )
    return models, skipped


# --- Metrics -------------------------------------------------------------------


def _errors(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Return ``predicted - actual`` after checking the shapes match."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape:
        raise ValueError(f"shape mismatch: {actual.shape} vs {predicted.shape}")
    errors: np.ndarray = predicted - actual
    return errors


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean absolute error; NaN for empty input."""
    errors = _errors(actual, predicted)
    return float(np.mean(np.abs(errors))) if errors.size else math.nan


def mse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean squared error; NaN for empty input."""
    errors = _errors(actual, predicted)
    return float(np.mean(errors**2)) if errors.size else math.nan


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root mean squared error; NaN for empty input."""
    return math.sqrt(mse(actual, predicted))


@dataclass
class HorizonScore:
    """Per-day errors for one model at one horizon.

    Aggregated like EpexPredictor: MAE is the mean of the daily MAEs and RMSE
    is the square root of the mean daily MSE.
    """

    daily_mae: list[float] = field(default_factory=list)
    daily_mse: list[float] = field(default_factory=list)

    def add_day(self, actual: np.ndarray, predicted: np.ndarray) -> None:
        """Score one forecast day, ignoring slots without an actual price."""
        known = ~np.isnan(actual)
        if not known.any():
            return
        self.daily_mae.append(mae(actual[known], predicted[known]))
        self.daily_mse.append(mse(actual[known], predicted[known]))

    def summary(self) -> tuple[float, float]:
        """Return (MAE, RMSE) over all scored days, NaN if none."""
        if not self.daily_mae:
            return math.nan, math.nan
        return float(np.mean(self.daily_mae)), math.sqrt(np.mean(self.daily_mse))


# --- Rolling backtest ------------------------------------------------------------


@dataclass(frozen=True)
class BacktestConfig:
    """Rolling-origin settings. Origins are local calendar days in ``tz``."""

    tz: tzinfo
    first_origin: date
    last_origin: date
    window_days: int = 180
    horizon_days: int = 3
    step_days: int = 1
    # Only these target days are scored (e.g. holidays, #26); None scores all
    score_days: frozenset[date] | None = None

    def origins(self) -> Iterator[date]:
        """Yield each forecast origin (the first forecast day)."""
        origin = self.first_origin
        while origin <= self.last_origin:
            yield origin
            origin += timedelta(days=self.step_days)


@dataclass
class BacktestResult:
    """Scores per model name, one ``HorizonScore`` per horizon day."""

    config: BacktestConfig
    scores: dict[str, list[HorizonScore]]
    origins: int = 0
    skipped_models: dict[str, str] = field(default_factory=dict)


def local_midnight(day: date, tz: tzinfo) -> int:
    """Return local midnight at the start of ``day`` in UTC unix seconds."""
    return int(datetime(day.year, day.month, day.day, tzinfo=tz).timestamp())


def history_for(
    series: PriceSeries, origin: date, config: BacktestConfig
) -> PriceSeries:
    """Return the only prices a model may see when forecasting from ``origin``.

    The horizon cutoff is local midnight at the start of ``origin``; the
    window reaches back ``config.window_days`` local days from there.
    """
    window_start = origin - timedelta(days=config.window_days)
    return series.between(
        local_midnight(window_start, config.tz), local_midnight(origin, config.tz)
    )


def target_slots(origin: date, config: BacktestConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return the slot starts to forecast and each slot's horizon day (1-based).

    The grid comes from the calendar rather than the data, so its size (92, 96
    or 100 slots on DST days) never reveals what exists after the cutoff.
    """
    bounds = np.array(
        [
            local_midnight(origin + timedelta(days=day), config.tz)
            for day in range(config.horizon_days + 1)
        ]
    )
    starts = np.arange(bounds[0], bounds[-1], SLOT_SECONDS, dtype=np.int64)
    horizon = np.searchsorted(bounds[1:], starts, side="right") + 1
    return starts, horizon


def holiday_days(region: str, first: date, last: date) -> frozenset[date]:
    """Return the days from ``first`` to ``last`` with a holiday that is not a Sunday.

    Sundays are holidays for the model (``public_holiday``), but they are
    ordinary Sundays for scoring.
    """
    days = (first + timedelta(days=n) for n in range((last - first).days + 1))
    return frozenset(
        day for day in days if day.weekday() != 6 and public_holiday(day, region)
    )


def run_backtest(
    series: PriceSeries,
    models: Sequence[Forecaster],
    config: BacktestConfig,
    progress: Callable[[date, int], None] | None = None,
) -> BacktestResult:
    """Retrain and forecast from every origin, scoring each horizon day.

    Only this function holds the full series; models get ``history_for`` and
    the target slot starts, never the prices being scored.
    """
    result = BacktestResult(
        config=config,
        scores={
            model.name: [HorizonScore() for _ in range(config.horizon_days)]
            for model in models
        },
    )
    for origin in config.origins():
        if config.score_days is not None and not any(
            origin + timedelta(days=day) in config.score_days
            for day in range(config.horizon_days)
        ):
            continue
        history = history_for(series, origin, config)
        if len(history) < MIN_HISTORY_SLOTS:
            continue
        targets, horizon = target_slots(origin, config)
        actual = series.prices_at(targets)
        if np.isnan(actual).all():
            continue
        for model in models:
            predicted = np.asarray(model.forecast(history, targets), dtype=float)
            if predicted.shape != targets.shape:
                raise ValueError(
                    f"{model.name} returned {predicted.shape}, expected {targets.shape}"
                )
            for day, score in enumerate(result.scores[model.name], start=1):
                if config.score_days is not None and (
                    origin + timedelta(days=day - 1) not in config.score_days
                ):
                    continue
                in_day = horizon == day
                score.add_day(actual[in_day], predicted[in_day])
        result.origins += 1
        if progress is not None:
            progress(origin, result.origins)
    return result


def format_report(result: BacktestResult, region: str) -> str:
    """Render the per-horizon MAE/RMSE table as markdown (EUR ct/kWh)."""
    config = result.config
    days = range(1, config.horizon_days + 1)
    header = ["Model"] + [f"{d}d {m}" for d in days for m in ("MAE", "RMSE")]
    lines = [
        f"Rolling backtest, {region}: origins {config.first_origin} to "
        f"{config.last_origin} ({result.origins} scored), "
        f"{config.window_days}-day window, retrained every {config.step_days} day(s). "
        + (
            f"Scored on {len(config.score_days)} holiday target days only. "
            if config.score_days is not None
            else ""
        )
        + "MAE/RMSE in EUR ct/kWh.",
        "",
        "| " + " | ".join(header) + " |",
        "| --- |" + " ---: |" * (len(header) - 1),
    ]
    for name, horizons in result.scores.items():
        cells = [f"{value:.2f}" for score in horizons for value in score.summary()]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    for name in result.skipped_models:
        lines.append(f"| {name} | " + " | ".join(["n/a"] * (len(header) - 1)) + " |")
    lines.extend(
        f"\n{name}: skipped, {reason}."
        for name, reason in result.skipped_models.items()
    )
    return "\n".join(lines)


# --- CLI -------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="python -m scripts.backtest",
        description="Rolling 1/2/3-day-ahead backtest of the OSF price model.",
    )
    parser.add_argument("--region", default="DK1", choices=sorted(ENERGY_CHARTS_ZONES))
    parser.add_argument(
        "--start",
        type=date.fromisoformat,
        help="first forecast origin, YYYY-MM-DD (default: 364 days before --end)",
    )
    parser.add_argument(
        "--end",
        type=date.fromisoformat,
        help="last forecast origin (default: 3 days ago, so D+3 has settled)",
    )
    parser.add_argument("--window-days", type=int, default=180)
    parser.add_argument(
        "--step-days", type=int, default=1, help="days between origins/retrains"
    )
    parser.add_argument(
        "--models",
        default=",".join(MODEL_NAMES),
        help=f"comma-separated subset of {','.join(MODEL_NAMES)}",
    )
    parser.add_argument(
        "--weather",
        choices=WEATHER_CHOICES,
        default="openmeteo",
        help="zone weather inputs: Open-Meteo's archived forecasts, or none",
    )
    parser.add_argument(
        "--horizon-days",
        type=int,
        default=3,
        help="forecast days scored per origin (1d, 2d, ...); 7 covers days 3-7",
    )
    parser.add_argument(
        "--load",
        choices=("none", "entsoe"),
        default="none",
        help="ENTSO-E's week-ahead load forecast (#30); needs ENTSOE_API_KEY",
    )
    parser.add_argument(
        "--gas",
        action="store_true",
        help="the daily gas price feature (#28), from Instrat",
    )
    parser.add_argument(
        "--cross-border",
        action="store_true",
        help="two-stage model with the neighbouring zones' prices (#29)",
    )
    parser.add_argument(
        "--days",
        choices=("all", "holidays"),
        default="all",
        help="score every target day, or only public holidays that are not Sundays",
    )
    parser.add_argument(
        "--nordpool-db",
        type=Path,
        help="learning-DB export (e.g. .cache/live/*.db): its nordpool_prognoses "
        "for training rows, none for target rows (#91)",
    )
    parser.add_argument(
        "--nordpool-day1",
        action="store_true",
        help="with --nordpool-db: the first forecast day has its prognosis too",
    )
    parser.add_argument(
        "--nordpool-no-copies",
        action="store_true",
        help="with --nordpool-db: no Nordpool-masked training copies (before #91)",
    )
    parser.add_argument(
        "--lags",
        help="lagged price columns (#119, tested and not kept): 'all' or a "
        "comma-separated subset of price_same_slot_last_known_day, "
        "price_mean_last_7_known_days, price_same_slot_last_week, price_lag_days",
    )
    parser.add_argument(
        "--lag-ages",
        action="store_true",
        help="with --lags: training rows lag 1-7 days (hashed per day) instead of 1",
    )
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/backtest"))
    parser.add_argument("--output", type=Path, help="also write the report here")
    args = parser.parse_args(argv)
    if args.window_days < 7 or args.step_days < 1 or args.horizon_days < 1:
        parser.error("--window-days must be >= 7, --step-days and --horizon-days >= 1")
    if args.load == "entsoe" and not os.environ.get("ENTSOE_API_KEY"):
        parser.error("--load entsoe needs the ENTSO-E token in ENTSOE_API_KEY")
    if args.cross_border and args.region not in NEIGHBOURS:
        parser.error(f"--cross-border needs a region in {', '.join(NEIGHBOURS)}")
    if (args.nordpool_day1 or args.nordpool_no_copies) and args.nordpool_db is None:
        parser.error("--nordpool-day1 and --nordpool-no-copies need --nordpool-db")
    if args.nordpool_db is not None and not args.nordpool_db.is_file():
        parser.error(f"--nordpool-db: no file {args.nordpool_db}")
    if args.lag_ages and args.lags is None:
        parser.error("--lag-ages needs --lags")
    if args.lags is not None:
        try:
            args.lags = LagConfig.parse(args.lags, args.lag_ages)
        except ValueError as err:
            parser.error(str(err))
    return args


def main(argv: Sequence[str] | None = None) -> int:
    """Run the backtest and print the markdown report."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    tz = ZoneInfo(str(REGIONS[args.region]["tz"]))
    last = args.end or date.today() - timedelta(days=3)
    first = args.start or last - timedelta(days=364)
    if first > last:
        raise SystemExit("--start must not be after --end")
    config = BacktestConfig(
        tz=tz,
        first_origin=first,
        last_origin=last,
        window_days=args.window_days,
        horizon_days=args.horizon_days,
        step_days=args.step_days,
        score_days=holiday_days(
            args.region, first, last + timedelta(days=args.horizon_days - 1)
        )
        if args.days == "holidays"
        else None,
    )
    names = [name.strip() for name in args.models.split(",") if name.strip()]
    data_first = first - timedelta(days=config.window_days + 1)
    data_last = last + timedelta(days=config.horizon_days)
    zone = None
    if args.weather == "openmeteo":
        print(f"Loading {args.region} weather from Open-Meteo ...", file=sys.stderr)
        zone = load_open_meteo_weather(
            args.region,
            data_first - timedelta(days=1),
            data_last + timedelta(days=1),
            args.cache_dir,
        )
    load = None
    if args.load == "entsoe":
        print(f"Loading {args.region} load forecast from ENTSO-E ...", file=sys.stderr)
        load = load_entsoe_load(
            args.region,
            data_first - timedelta(days=1),
            data_last + timedelta(days=1),
            args.cache_dir,
            os.environ["ENTSOE_API_KEY"],
        )
    cross = None
    if args.cross_border:
        cross = load_cross_border(
            args.region,
            data_first,
            data_last,
            args.cache_dir,
            with_weather=args.weather == "openmeteo",
        )
    gas = None
    if args.gas:
        print("Loading gas prices from Instrat ...", file=sys.stderr)
        gas = load_gas_prices(
            data_first - timedelta(days=GAS_LOOKBACK_DAYS), data_last, args.cache_dir
        )
    nordpool = None
    if args.nordpool_db is not None:
        nordpool = load_nordpool_db(
            args.nordpool_db,
            day1=args.nordpool_day1,
            copies=not args.nordpool_no_copies,
        )
    models, skipped = build_models(
        names, tz, zone, args.region, load, cross, gas, nordpool, args.lags
    )

    print(f"Loading {args.region} prices from energy-charts ...", file=sys.stderr)
    series = load_energy_charts_prices(
        args.region, data_first, data_last, args.cache_dir
    )
    started = time.monotonic()

    def progress(origin: date, scored: int) -> None:
        if scored == 1 or scored % 10 == 0:
            elapsed = time.monotonic() - started
            print(f"  {origin}: {scored} origins ({elapsed:.0f} s)", file=sys.stderr)

    result = run_backtest(series, models, config, progress)
    result.skipped_models.update(skipped)
    report = format_report(result, args.region)
    print(report)
    if args.output is not None:
        args.output.write_text(report + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
