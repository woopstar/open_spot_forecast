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
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, tzinfo
from pathlib import Path
from typing import Any, Protocol
from zoneinfo import ZoneInfo

import numpy as np

from custom_components.open_spot_forecast.api.dayahead_prices import (
    parse_energy_charts as parse_dayahead_rows,
)
from custom_components.open_spot_forecast.api.openmeteo_weather import (
    open_meteo_query,
    parse_open_meteo,
)
from custom_components.open_spot_forecast.const import (
    ENERGY_CHARTS_API,
    OPEN_METEO_ARCHIVE_API,
    REGIONS,
    WEATHER_POINTS,
)
from custom_components.open_spot_forecast.ml.features import (
    FEATURE_NAMES,
    SlotInputs,
    build_feature_row,
    build_feature_vector,
)
from custom_components.open_spot_forecast.ml.gbm import NumpyGradientBoosting
from custom_components.open_spot_forecast.ml.models import create_price_model
from custom_components.open_spot_forecast.ml.zone_weather import ZoneWeatherIndex

SLOT_SECONDS = 15 * 60
# Origins with less history than this are skipped (the naive model needs a week).
MIN_HISTORY_SLOTS = 7 * 96

HTTP_ATTEMPTS = 5
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
    """Fetch and decode a JSON document, waiting out HTTP 429 rate limits."""
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
            if err.code != 429 or attempt >= HTTP_ATTEMPTS:
                raise
            wait = _retry_after_seconds(err.headers.get("Retry-After"))
            print(f"  rate limited, retrying in {wait:.0f} s", file=sys.stderr)
            sleep(wait)
            attempt += 1


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
    Prices are © Bundesnetzagentur | SMARD.de, CC BY 4.0, via energy-charts.info.
    """
    zone = ENERGY_CHARTS_ZONES[region]
    today = today or date.today()
    parts = []
    for month_start, month_end in _month_chunks(first, last):
        cache_file = cache_dir / f"energy_charts_{zone}_{month_start:%Y-%m}.json"
        if cache_file.exists():
            payload = json.loads(cache_file.read_text(encoding="utf-8"))
        else:
            query = urllib.parse.urlencode(
                {
                    "bzn": zone,
                    "start": month_start.isoformat(),
                    "end": month_end.isoformat(),
                }
            )
            payload = fetch(f"{ENERGY_CHARTS_API}?{query}")
            if month_end < today:
                cache_dir.mkdir(parents=True, exist_ok=True)
                cache_file.write_text(json.dumps(payload), encoding="utf-8")
        parts.append(parse_energy_charts(payload))
    return merge_series(parts)


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
        if cache_file.exists():
            payload = json.loads(cache_file.read_text(encoding="utf-8"))
        else:
            query = urllib.parse.urlencode(
                open_meteo_query(points, chunk_start, chunk_end)
            )
            payload = fetch(f"{OPEN_METEO_ARCHIVE_API}?{query}")
            if chunk_end < today - timedelta(days=2):
                cache_dir.mkdir(parents=True, exist_ok=True)
                cache_file.write_text(json.dumps(payload), encoding="utf-8")
        rows.extend(parse_open_meteo(payload, points))
        chunk_start = chunk_end + timedelta(days=1)
    return ZoneWeatherIndex(rows)


# --- Models --------------------------------------------------------------------


def feature_matrix(
    starts: np.ndarray, tz: tzinfo, zone: ZoneWeatherIndex | None = None
) -> np.ndarray:
    """Return the integration's model input for each slot start.

    Rows come from ``build_feature_row``, as in training and prediction. The
    zone weather (#22) comes from Open-Meteo's archived forecasts, for
    training and target slots alike. The local weather entity and Nordpool
    inputs have no year of history, so they are unknown (NaN).
    """
    rows = []
    for start in starts.tolist():
        moment = datetime.fromtimestamp(start, tz)
        inputs = SlotInputs(**zone.for_slot(moment)) if zone else SlotInputs()
        rows.append(build_feature_vector(build_feature_row(moment, inputs)))
    return np.array(rows, dtype=float).reshape(len(rows), len(FEATURE_NAMES))


def _feature_rows(
    history: PriceSeries,
    targets: np.ndarray,
    tz: tzinfo,
    zone: ZoneWeatherIndex | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (training rows, target rows) for the slot times and zone weather."""
    return feature_matrix(history.starts, tz, zone), feature_matrix(targets, tz, zone)


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

    def forecast(self, history: PriceSeries, targets: np.ndarray) -> np.ndarray:
        """Fit a fresh model on ``history`` and predict the targets."""
        train_rows, target_rows = _feature_rows(history, targets, self.tz, self.zone)
        model = self.model_factory()
        model.fit(train_rows, history.prices)
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

    def forecast(self, history: PriceSeries, targets: np.ndarray) -> np.ndarray:
        """Fit a fresh LightGBM booster on ``history`` and predict the targets."""
        import lightgbm  # optional dev-only dependency (requirements_backtest.txt)

        train_rows, target_rows = _feature_rows(history, targets, self.tz, self.zone)
        dataset = lightgbm.Dataset(
            train_rows, label=history.prices, feature_name=list(FEATURE_NAMES)
        )
        booster = lightgbm.train(
            LIGHTGBM_PARAMS, dataset, num_boost_round=LIGHTGBM_ROUNDS
        )
        return np.asarray(booster.predict(target_rows), dtype=float)


def _lightgbm_available() -> bool:
    """Return True if the optional lightgbm package is importable."""
    return importlib.util.find_spec("lightgbm") is not None


def build_models(
    names: Sequence[str], tz: tzinfo, zone: ZoneWeatherIndex | None = None
) -> tuple[list[Forecaster], dict[str, str]]:
    """Instantiate the requested models; return them plus {skipped name: reason}."""
    models: list[Forecaster] = []
    skipped: dict[str, str] = {}
    for name in names:
        if name == "naive":
            models.append(NaiveLastWeek(tz))
        elif name == "current":
            models.append(CurrentModel(tz, zone=zone))
        elif name == "lightgbm":
            if _lightgbm_available():
                models.append(LightGbmReference(tz, zone=zone))
            else:
                skipped[LightGbmReference.name] = (
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
        "MAE/RMSE in EUR ct/kWh.",
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
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/backtest"))
    parser.add_argument("--output", type=Path, help="also write the report here")
    args = parser.parse_args(argv)
    if args.window_days < 7 or args.step_days < 1:
        parser.error("--window-days must be >= 7 and --step-days >= 1")
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
        step_days=args.step_days,
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
    models, skipped = build_models(names, tz, zone)

    print(f"Loading {args.region} prices from energy-charts ...", file=sys.stderr)
    series = load_energy_charts_prices(
        args.region, data_first, data_last, args.cache_dir
    )
    started = time.monotonic()

    def progress(origin: date, scored: int) -> None:
        if scored % 10 == 0:
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
