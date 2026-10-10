"""Benchmark Open Spot Forecast against external price forecasts (dev-only, #115).

Answers the question a Danish user has: is OSF's forecast better or worse than
the forecasts available for free (Smartere Elforbrug, Carnot, EpexPredictor)?
Two sub-commands, run from the repository root::

    python -m scripts.benchmark collect --region DK1
    python -m scripts.benchmark collect --region DK1 --sources smartere --backfill-days 60
    python -m scripts.benchmark report --region DK1

``collect`` snapshots every source's forecast as published, OSF's included,
into ``.cache/benchmark/<region>.db`` (``scripts/benchmark_sources.py``; run it
once or twice a day, before the day-ahead auction). ``report`` scores them
against energy-charts' day-ahead prices with the backtest's metric definitions.

Two rules keep the comparison honest:

- **Origin.** A forecast for local day *T* is scored at lead *k* only if it was
  published before ``--cutoff`` (12:00 local) on day *T - k*: for *k = 1* that
  is before the auction result for *T*, after which every source, OSF
  included, simply repeats the known prices. The latest such forecast of each
  source is used.
- **Resolution.** Every source is averaged to hours before it is compared with
  the hourly mean of the actual prices, so a 15-minute source gets no
  advantage or penalty against an hourly one.

Not shipped with the integration; raw forecasts never leave ``.cache/``.
"""

import argparse
import bisect
import math
import os
import sys
import urllib.error
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, tzinfo
from pathlib import Path
from typing import TextIO

import numpy as np

from custom_components.open_spot_forecast.api.exchange_rates import PEGGED_EUR_RATES

from .backtest import (
    CT_PER_KWH_PER_EUR_PER_MWH,
    SLOT_SECONDS,
    HorizonScore,
    NaiveLastWeek,
    PriceSeries,
    load_energy_charts_prices,
    local_midnight,
)
from .benchmark_sources import (
    BENCHMARK_REGIONS,
    SOURCES,
    BenchmarkStore,
    FetchJson,
    Forecast,
    SourceUnavailable,
    backfill_smartere,
    collect_carnot,
    collect_epex,
    collect_osf,
    collect_smartere,
    default_store_path,
    fetch_json,
    load_osf_export,
    origin_time,
)
from .live_report import Units, _table, region_tz

HOUR_SECONDS = 3600
LEADS = tuple(range(1, 8))
# The window of the "find the cheap hours" metric
CHEAP_WINDOW_HOURS = 3
NAIVE = "naive (same slot last week)"
# Per-source forecasts: {source: {origin: {slot start: EUR/MWh}}}
Collected = Mapping[str, Mapping[int, Mapping[int, float]]]


# --- Collect ----------------------------------------------------------------------


@dataclass(frozen=True)
class CollectOptions:
    """What ``collect`` fetches."""

    region: str
    sources: Sequence[str]
    backfill_days: int = 0
    osf_db: Path | None = None
    currency: str = "DKK"


def _failure(err: Exception) -> str:
    """Describe a failed request without its URL (which may carry a host or key)."""
    if isinstance(err, urllib.error.HTTPError):
        return f"HTTP {err.code}"
    if isinstance(err, urllib.error.URLError):
        return f"connection failed ({err.reason})"
    return f"{type(err).__name__}: {err}"


def _collect_source(
    source: str,
    options: CollectOptions,
    store: BenchmarkStore,
    now: datetime,
    fetch: FetchJson,
    environ: Mapping[str, str],
    out: TextIO,
) -> list[Forecast]:
    """Return a source's forecast(s) to store; a backfill stores as it goes."""
    region = options.region
    if source == "smartere":
        if options.backfill_days > 0:
            since = now - timedelta(days=options.backfill_days)
            stored, stopped = backfill_smartere(store, region, since, fetch, environ)
            print(f"smartere: {stored} past forecast(s) backfilled", file=out)
            if stopped:
                print(f"smartere: backfill stopped, {stopped}", file=out)
        return [collect_smartere(region, fetch)]
    if source == "carnot":
        return [collect_carnot(region, now, fetch, environ)]
    if source == "epex":
        return [collect_epex(region, now, fetch)]
    if options.osf_db is not None:
        return load_osf_export(options.osf_db, options.currency, region_tz(region))
    return [collect_osf(now, fetch, environ)]


def collect(
    options: CollectOptions,
    store: BenchmarkStore,
    now: datetime,
    fetch: FetchJson = fetch_json,
    environ: Mapping[str, str] = os.environ,
    out: TextIO | None = None,
) -> int:
    """Store every selected source's forecast as published now.

    A source without credentials is skipped and a failing one does not stop
    the others. Storing is idempotent: a forecast already stored (the same
    source, origin and slot) adds nothing, and a source fetched again before
    its forecast changed is not stored a second time.

    Returns:
        0, or 1 if every selected source failed.
    """
    out = out or sys.stdout
    failed = 0
    for source in options.sources:
        try:
            forecasts = _collect_source(
                source, options, store, now, fetch, environ, out
            )
        except SourceUnavailable as err:
            print(f"{source}: skipped, {err}", file=out)
            continue
        except (OSError, ValueError, KeyError, TypeError) as err:
            # OSError covers urllib's errors, timeouts and a missing export
            print(f"{source}: failed, {_failure(err)}", file=out)
            failed += 1
            continue
        fetched = int(now.timestamp())
        for forecast in forecasts:
            # Only a forecast without a publication time of its own
            seen = (
                store.unchanged_since(forecast) if forecast.origin == fetched else None
            )
            if seen is not None:
                print(f"{source}: unchanged since {origin_time(seen)}", file=out)
                continue
            new = store.add(forecast)
            print(
                f"{source}: {len(forecast.points)} prices published "
                f"{origin_time(forecast.origin)} ({new} new)",
                file=out,
            )
    return 1 if options.sources and failed == len(options.sources) else 0


# --- Report: selection and alignment ----------------------------------------------


@dataclass(frozen=True)
class ReportConfig:
    """How forecasts are picked and compared."""

    tz: tzinfo
    cutoff: time = time(12, 0)
    leads: Sequence[int] = LEADS
    since: date | None = None


def cutoff_seconds(target: date, lead: int, config: ReportConfig) -> int:
    """Return the latest origin (exclusive) a forecast for ``target`` may have.

    That is ``config.cutoff`` local time on the day ``lead`` days before the
    target day, in UTC unix seconds.
    """
    day = target - timedelta(days=lead)
    return int(datetime.combine(day, config.cutoff, tzinfo=config.tz).timestamp())


def select_origin(origins: Sequence[int], cutoff: int) -> int | None:
    """Return the latest origin strictly before ``cutoff``, None without one.

    Args:
        origins: A source's origins, sorted ascending.
        cutoff: UTC unix seconds; a forecast published at or after it is
            never used.
    """
    index = bisect.bisect_left(origins, cutoff)
    return origins[index - 1] if index else None


def hourly_means(points: Mapping[int, float]) -> dict[int, float]:
    """Average prices per UTC hour (hourly prices are returned as they are)."""
    hours: dict[int, list[float]] = {}
    for start, price in points.items():
        hours.setdefault(start - start % HOUR_SECONDS, []).append(price)
    return {hour: sum(prices) / len(prices) for hour, prices in hours.items()}


def day_hours(day: date, tz: tzinfo) -> np.ndarray:
    """Return the hour starts of a local day (23, 24 or 25 on DST days)."""
    return np.arange(
        local_midnight(day, tz),
        local_midnight(day + timedelta(days=1), tz),
        HOUR_SECONDS,
        dtype=np.int64,
    )


def _hourly(quarters: np.ndarray) -> np.ndarray:
    """Return the mean of each hour's known 15-minute prices, NaN without one."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        means: np.ndarray = np.nanmean(quarters.reshape(-1, 4), axis=1)
    return means


def _quarters(hours: np.ndarray) -> np.ndarray:
    """Return the four 15-minute slot starts of every hour, in order."""
    slots: np.ndarray = (
        hours[:, None] + np.arange(0, HOUR_SECONDS, SLOT_SECONDS)[None, :]
    ).reshape(-1)
    return slots


def actual_hourly(series: PriceSeries, hours: np.ndarray) -> np.ndarray:
    """Return the actual price per hour (EUR ct/kWh): the mean of its slots."""
    return _hourly(series.prices_at(_quarters(hours)))


def naive_hourly(series: PriceSeries, day: date, tz: tzinfo) -> np.ndarray:
    """Return the backtest's naive forecast for a day, per hour.

    The same local slot a week earlier (``NaiveLastWeek``), from the prices
    of the eight days before the target day: all known at every lead's
    cutoff, so the row is the same floor at every lead.
    """
    history = series.between(
        local_midnight(day - timedelta(days=8), tz), local_midnight(day, tz)
    )
    hours = day_hours(day, tz)
    if not len(history):
        return np.full(hours.shape, math.nan)
    return _hourly(NaiveLastWeek(tz).forecast(history, _quarters(hours)))


def forecast_hourly(
    points: Mapping[int, float], hours: np.ndarray
) -> np.ndarray | None:
    """Return a forecast per hour of a day in EUR ct/kWh; None if it lacks an hour."""
    means = hourly_means(points)
    if any(int(hour) not in means for hour in hours):
        return None
    return np.array([means[int(hour)] for hour in hours]) * CT_PER_KWH_PER_EUR_PER_MWH


# --- Report: metrics --------------------------------------------------------------


def cheapest_window(prices: np.ndarray, hours: int = CHEAP_WINDOW_HOURS) -> int:
    """Return the start index of the cheapest run of ``hours`` consecutive prices."""
    return int(np.argmin(np.convolve(prices, np.ones(hours), mode="valid")))


def _ranks(values: np.ndarray) -> np.ndarray:
    """Return 1-based ranks, tied values sharing their mean rank."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size)
    ranks[order] = np.arange(1, values.size + 1)
    for value in np.unique(values):
        tied = values == value
        ranks[tied] = ranks[tied].mean()
    return ranks


def rank_correlation(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Return Spearman's rank correlation; NaN when either side is constant."""
    a, b = _ranks(actual), _ranks(predicted)
    spread = float(np.std(a) * np.std(b))
    if spread < 1e-12:
        return math.nan
    return float(np.mean((a - a.mean()) * (b - b.mean())) / spread)


@dataclass
class LeadScore:
    """One source's errors at one lead, a day at a time.

    MAE and RMSE are the backtest's (``HorizonScore``): the mean of the daily
    MAEs and the root of the mean daily MSE. Bias is the mean of the daily
    mean errors (forecast - actual).
    """

    errors: HorizonScore = field(default_factory=HorizonScore)
    daily_bias: list[float] = field(default_factory=list)
    overlaps: list[bool] = field(default_factory=list)
    correlations: list[float] = field(default_factory=list)

    @property
    def days(self) -> int:
        """Return the number of scored days."""
        return len(self.daily_bias)

    def add_day(self, actual: np.ndarray, predicted: np.ndarray) -> None:
        """Score one target day of hourly prices (EUR ct/kWh)."""
        self.errors.add_day(actual, predicted)
        self.daily_bias.append(float(np.mean(predicted - actual)))
        if actual.size >= CHEAP_WINDOW_HOURS:
            distance = abs(cheapest_window(predicted) - cheapest_window(actual))
            self.overlaps.append(distance < CHEAP_WINDOW_HOURS)
        correlation = rank_correlation(actual, predicted)
        if not math.isnan(correlation):
            self.correlations.append(correlation)

    def bias(self) -> float:
        """Return the mean daily bias, NaN without days."""
        return float(np.mean(self.daily_bias)) if self.daily_bias else math.nan

    def overlap_share(self) -> float:
        """Return the share of days whose cheapest windows overlap."""
        return float(np.mean(self.overlaps)) if self.overlaps else math.nan

    def correlation(self) -> float:
        """Return the mean daily rank correlation."""
        return float(np.mean(self.correlations)) if self.correlations else math.nan


# {source: {lead: score}}
Scores = dict[str, dict[int, LeadScore]]


@dataclass
class BenchmarkResult:
    """The scores over every day a source has, and over the days all have."""

    sources: Sequence[str]
    available: Scores
    paired: Scores
    first_day: date | None = None
    last_day: date | None = None


def _target_days(
    collected: Collected, series: PriceSeries, config: ReportConfig
) -> list[date]:
    """Return the local days that have both a forecast slot and actual prices."""
    starts = [
        start
        for origins in collected.values()
        for points in origins.values()
        for start in points
    ]
    if not starts or not len(series):
        return []
    first = datetime.fromtimestamp(min(starts), config.tz).date()
    last = datetime.fromtimestamp(
        min(max(starts), int(series.starts[-1])), config.tz
    ).date()
    if config.since is not None:
        first = max(first, config.since)
    return [first + timedelta(days=n) for n in range((last - first).days + 1)]


def run_benchmark(
    collected: Collected,
    series: PriceSeries,
    sources: Sequence[str],
    config: ReportConfig,
) -> BenchmarkResult:
    """Score every source per lead against the actual prices.

    Args:
        collected: The stored forecasts (``BenchmarkStore.forecasts``).
        series: The actual day-ahead prices (EUR ct/kWh, 15-minute grid).
        sources: The sources to report; the naive row is always added.
        config: The cutoff rule, the leads and the first target day.
    """
    names = [*sources, NAIVE]
    result = BenchmarkResult(
        sources=names,
        available={name: {k: LeadScore() for k in config.leads} for name in names},
        paired={name: {k: LeadScore() for k in config.leads} for name in names},
    )
    origins = {source: sorted(collected.get(source, {})) for source in sources}
    for day in _target_days(collected, series, config):
        hours = day_hours(day, config.tz)
        actual = actual_hourly(series, hours)
        # A day is scored whole: missing prices would leave out its peak or trough
        if np.isnan(actual).any():
            continue
        naive = naive_hourly(series, day, config.tz)
        scored = False
        for lead in config.leads:
            cutoff = cutoff_seconds(day, lead, config)
            predicted: dict[str, np.ndarray] = {}
            for source in sources:
                origin = select_origin(origins[source], cutoff)
                if origin is None:
                    continue
                values = forecast_hourly(collected[source][origin], hours)
                if values is not None:
                    predicted[source] = values
            if not predicted:
                continue
            scored = True
            if not np.isnan(naive).any():
                predicted[NAIVE] = naive
            everyone = all(source in predicted for source in sources)
            for name, values in predicted.items():
                result.available[name][lead].add_day(actual, values)
                if everyone:
                    result.paired[name][lead].add_day(actual, values)
        if scored:
            result.first_day = result.first_day or day
            result.last_day = day
    return result


# --- Report: output ---------------------------------------------------------------


def _error_rows(scores: Scores, names: Sequence[str], units: Units) -> list[list[str]]:
    rows = []
    for lead in LEADS:
        for name in names:
            score = scores[name].get(lead)
            if score is None or not score.days:
                continue
            mae, rmse = score.errors.summary()
            rows.append(
                [
                    f"{lead}d",
                    name,
                    str(score.days),
                    units.fmt(mae),
                    units.fmt(rmse),
                    units.fmt(score.bias()),
                ]
            )
    return rows


def _minima_rows(scores: Scores, names: Sequence[str]) -> list[list[str]]:
    rows = []
    for lead in LEADS:
        for name in names:
            score = scores[name].get(lead)
            if score is None or not score.days:
                continue
            rows.append(
                [
                    f"{lead}d",
                    name,
                    str(score.days),
                    f"{score.overlap_share():.0%}",
                    f"{score.correlation():.2f}",
                ]
            )
    return rows


def format_report(
    result: BenchmarkResult, region: str, config: ReportConfig, units: Units
) -> str:
    """Render the benchmark as markdown."""
    lines = [f"# Benchmark against external forecasts: {region}", ""]
    if result.first_day is None:
        lines += [
            "No forecast can be scored yet: collect forecasts on several days "
            "(`python -m scripts.benchmark collect`), then report again.",
        ]
        return "\n".join(lines) + "\n"
    header = ["Lead", "Source", "Days", "MAE", "RMSE", "Bias"]
    lines += [
        f"Target days {result.first_day} to {result.last_day} (local). A forecast "
        f"is scored at lead *k* if it was published before {config.cutoff:%H:%M} "
        f"local time *k* days before the target day; the latest one is used. "
        f"Hourly prices, errors in {units.label}.",
        "",
        "## Paired days (every source has a forecast)",
        "",
    ]
    paired = _error_rows(result.paired, result.sources, units)
    lines += _table(header, paired) if paired else ["No day has every source."]
    lines += ["", "## All available days", ""]
    lines += _table(header, _error_rows(result.available, result.sources, units))
    lines += [
        "",
        "MAE is the mean of the daily MAEs and RMSE the root of the mean daily "
        "MSE, as in `scripts/backtest.py`; bias is the mean of forecast - actual. "
        "Days differ per source in the second table, so compare sources in the "
        "first one.",
        "",
        "## Finding the cheap hours (all available days)",
        "",
    ]
    lines += _table(
        [
            "Lead",
            "Source",
            "Days",
            f"Cheapest {CHEAP_WINDOW_HOURS} h overlap",
            "Rank correlation",
        ],
        _minima_rows(result.available, result.sources),
    )
    lines += [
        "",
        f"Overlap: the share of days whose forecast cheapest {CHEAP_WINDOW_HOURS}-hour "
        "window shares at least one hour with the actual one. Rank correlation: "
        "Spearman's, of the day's hourly prices, averaged over the days.",
    ]
    return "\n".join(lines) + "\n"


# --- CLI ----------------------------------------------------------------------------


def _sources(value: str) -> list[str]:
    names = [name.strip() for name in value.split(",") if name.strip()]
    unknown = sorted(set(names) - set(SOURCES))
    if unknown or not names:
        raise argparse.ArgumentTypeError(
            f"unknown source(s) {', '.join(unknown)}; choose from {', '.join(SOURCES)}"
        )
    return list(dict.fromkeys(names))


def _cutoff(value: str) -> time:
    try:
        return time.fromisoformat(value)
    except ValueError as err:
        raise argparse.ArgumentTypeError("expected HH:MM") from err


def report_units(currency: str) -> Units:
    """Return the report's unit: EUR ct/kWh, or another currency's hundredths."""
    if currency == "EUR":
        return Units(1.0, "EUR ct/kWh", 2)
    rate = PEGGED_EUR_RATES[currency]
    return Units(rate, f"{currency} 1/100 per kWh (1 EUR = {rate:g} {currency})", 2)


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="python -m scripts.benchmark",
        description="Benchmark Open Spot Forecast against external price forecasts.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name, text in (
        ("collect", "store every source's forecast as published now"),
        ("report", "score the stored forecasts against the actual prices"),
    ):
        command = commands.add_parser(name, help=text, description=text)
        command.add_argument("--region", default="DK1", choices=BENCHMARK_REGIONS)
        command.add_argument(
            "--sources",
            "--source",
            type=_sources,
            default=list(SOURCES),
            help=f"comma-separated, from {', '.join(SOURCES)} (default: all)",
        )
        command.add_argument(
            "--cache-dir",
            type=Path,
            default=Path(".cache"),
            help="holds benchmark/<region>.db and the cached actual prices",
        )
        if name == "collect":
            command.add_argument(
                "--backfill-days",
                type=int,
                default=0,
                help="also reconstruct Smartere Elforbrug's forecasts of the "
                "last N days from its git history",
            )
            command.add_argument(
                "--osf-db",
                type=Path,
                help="read OSF's pending predictions from a learning-database "
                "export instead of a running instance (HA_URL, HA_TOKEN)",
            )
            command.add_argument(
                "--currency",
                default="DKK",
                choices=["EUR", *sorted(PEGGED_EUR_RATES)],
                help="currency of the --osf-db export's prices",
            )
        else:
            command.add_argument(
                "--since",
                type=date.fromisoformat,
                help="only target days from this local date on, YYYY-MM-DD",
            )
            command.add_argument(
                "--cutoff",
                type=_cutoff,
                default=time(12, 0),
                help="latest local publication time on day T-k (default 12:00, "
                "before the day-ahead auction result)",
            )
            command.add_argument(
                "--currency",
                default="EUR",
                choices=["EUR", *sorted(PEGGED_EUR_RATES)],
                help="report unit: EUR ct/kWh (default) or a pegged currency",
            )
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
    now: Callable[[], datetime] = lambda: datetime.now().astimezone(),
) -> int:
    """Run the ``collect`` or ``report`` sub-command."""
    args = parse_args(argv)
    store = BenchmarkStore(default_store_path(args.region, args.cache_dir))
    try:
        if args.command == "collect":
            options = CollectOptions(
                args.region,
                args.sources,
                args.backfill_days,
                args.osf_db,
                args.currency,
            )
            return collect(options, store, now(), fetch_json)
        tz = region_tz(args.region)
        config = ReportConfig(tz=tz, cutoff=args.cutoff, since=args.since)
        collected = store.forecasts(args.sources)
        starts = [
            start
            for origins in collected.values()
            for points in origins.values()
            for start in points
        ]
        series = PriceSeries.from_points([], [])
        if starts:
            # The naive row needs the eight days before the first target day
            first = datetime.fromtimestamp(min(starts), tz).date() - timedelta(days=9)
            last = min(datetime.fromtimestamp(max(starts), tz).date(), now().date())
            series = load_energy_charts_prices(
                args.region, first, last, args.cache_dir / "backtest"
            )
        result = run_benchmark(collected, series, args.sources, config)
        print(
            format_report(result, args.region, config, report_units(args.currency)),
            end="",
        )
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
