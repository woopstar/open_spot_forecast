"""Live forecast accuracy from a learning-database export (dev-only, #94).

The backtest (``scripts/backtest.py``) is optimistic from day 2 on: its target
rows get Open-Meteo's archived, near-same-day forecasts. The live numbers are
in the learning database of a running instance, which this script reads:

- ``lead_time_accuracy``: daily error sums per lead-time bucket (day 1/2/3/4+),
  kept for ``LEAD_TIME_WINDOW_DAYS``;
- ``evaluation``: per slot, the prediction closest to a day ahead next to the
  actual price, kept for ``EVALUATION_KEEP_DAYS``;
- ``meta``: the latest training's holdout MAE/RMSE (and the ``hpo_*`` keys of
  exports from before #92).

The export is opened read-only (``?immutable=1``): the report never migrates,
checkpoints or writes it. Prices are in the model's unit (raw spot price excl.
VAT, currency/kWh); ``--eur-per-unit`` or ``--currency`` converts them to the
backtest's EUR ct/kWh. Not shipped with the integration. Run it from the
repository root::

    python -m scripts.live_report .cache/live/dk1_live.db --currency DKK
    python -m scripts.live_report .cache/live/dk1_live.db --log home-assistant.log
"""

import argparse
import math
import re
import sqlite3
import statistics
from collections.abc import Sequence
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta, tzinfo
from pathlib import Path
from zoneinfo import ZoneInfo

from custom_components.open_spot_forecast.api.exchange_rates import PEGGED_EUR_RATES
from custom_components.open_spot_forecast.const import (
    EVALUATION_KEEP_DAYS,
    LEAD_TIME_BUCKETS,
    LEAD_TIME_WINDOW_DAYS,
    REGIONS,
)
from custom_components.open_spot_forecast.ml.lead_time import summarize_error_sums
from custom_components.open_spot_forecast.ml.models import OBSOLETE_HPO_META_KEYS
from custom_components.open_spot_forecast.time_slots import UTC_KEY_FORMAT

# (samples, sum_error, sum_abs_error, sum_sq_error), as in lead_time_accuracy
ErrorSums = tuple[int, float, float, float]

# Fewer days than this are too few to record as live accuracy in the docs
MIN_REPORT_DAYS = 14
# The log line ModelMixin._train_models writes after every training
TRAINING_LOG = re.compile(r"ML model trained in (\d+(?:\.\d+)?) s")
_BUCKET_ORDER = [bucket for bucket, _ in LEAD_TIME_BUCKETS]


@dataclass(frozen=True)
class ErrorStats:
    """MAE, RMSE and bias (mean signed error) of a set of prediction errors.

    ``mae``/``rmse``/``bias`` pool every sample, as the accuracy sensors do.
    ``daily_mae``/``daily_rmse`` are the mean daily MAE and the root of the
    mean daily MSE, the backtest's method, so the two compare directly.
    """

    samples: int
    days: int
    mae: float
    rmse: float
    bias: float
    daily_mae: float
    daily_rmse: float


@dataclass(frozen=True)
class EvaluationRow:
    """One ``evaluation`` row: a slot's day-ahead prediction and actual price."""

    timestamp: str
    predicted: float
    actual: float
    lead_hours: float


@dataclass(frozen=True)
class LiveData:
    """The tables of an export the report reads."""

    path: Path
    # (local slot date, bucket) -> error sums
    lead_time: dict[tuple[date, str], ErrorSums]
    evaluation: list[EvaluationRow]
    meta: dict[str, str]


@dataclass(frozen=True)
class Units:
    """How reported prices are scaled and labelled."""

    factor: float = 1.0
    label: str = "currency/kWh (model unit)"
    digits: int = 3

    def fmt(self, value: float | None) -> str:
        """Format a price (error) in the report's unit."""
        if value is None:
            return "–"
        return f"{value * self.factor:.{self.digits}f}"


def open_read_only(path: Path) -> sqlite3.Connection:
    """Open an export read-only, without touching it (``immutable=1``).

    ``immutable`` skips locking and never creates ``-wal``/``-shm`` files, so
    it only sees committed pages: export with ``sqlite3 … ".backup …"``, not
    by copying a live database's main file alone.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    if not path.is_file():
        raise FileNotFoundError(path)
    return sqlite3.connect(f"{path.resolve().as_uri()}?immutable=1", uri=True)


def load_live_data(path: Path, since: date | None = None, tz: tzinfo = UTC) -> LiveData:
    """Read the accuracy, evaluation and meta tables of an export.

    A table the export does not have (an older schema) reads as empty.

    Args:
        path: The export.
        since: Keep only slots from this local date on (e.g. the day after
            updating the instance), so an older version's errors are left out.
        tz: The region's time zone, for the evaluation rows' local dates.
    """
    with closing(open_read_only(path)) as conn:
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        lead_time: dict[tuple[date, str], ErrorSums] = {}
        if "lead_time_accuracy" in tables:
            for day, bucket, *sums in conn.execute(
                "SELECT date, bucket, samples, sum_error, sum_abs_error, sum_sq_error"
                " FROM lead_time_accuracy"
            ):
                slot_date = date.fromisoformat(day)
                if since is not None and slot_date < since:
                    continue
                lead_time[(slot_date, bucket)] = (
                    int(sums[0]),
                    float(sums[1]),
                    float(sums[2]),
                    float(sums[3]),
                )
        evaluation: list[EvaluationRow] = []
        if "evaluation" in tables:
            evaluation = [
                EvaluationRow(str(row[0]), float(row[1]), float(row[2]), float(row[3]))
                for row in conn.execute(
                    "SELECT timestamp, predicted, actual, lead_hours FROM evaluation"
                    " ORDER BY julianday(timestamp)"
                )
                if since is None or _local_date(str(row[0]), tz) >= since
            ]
        meta: dict[str, str] = {}
        if "meta" in tables:
            meta = {
                str(key): str(value)
                for key, value in conn.execute("SELECT key, value FROM meta")
            }
    return LiveData(path, lead_time, evaluation, meta)


def _local_date(utc_key: str, tz: tzinfo) -> date:
    """Return the local date of a UTC slot key (``utc_slot_key``)."""
    start = datetime.strptime(utc_key, UTC_KEY_FORMAT).replace(tzinfo=UTC)
    return start.astimezone(tz).date()


def error_stats(daily: Sequence[ErrorSums]) -> ErrorStats | None:
    """Combine daily error sums; None without samples."""
    days = [sums for sums in daily if sums[0] > 0]
    if not days:
        return None
    total: ErrorSums = (
        sum(sums[0] for sums in days),
        sum(sums[1] for sums in days),
        sum(sums[2] for sums in days),
        sum(sums[3] for sums in days),
    )
    pooled = summarize_error_sums({"all": total})["all"]
    return ErrorStats(
        samples=total[0],
        days=len(days),
        mae=float(pooled["mae"]),
        rmse=float(pooled["rmse"]),
        bias=float(pooled["bias"]),
        daily_mae=statistics.fmean(sums[2] / sums[0] for sums in days),
        daily_rmse=math.sqrt(statistics.fmean(sums[3] / sums[0] for sums in days)),
    )


def evaluation_stats(rows: Sequence[EvaluationRow], tz: tzinfo) -> ErrorStats | None:
    """Return the evaluation table's errors, grouped per local day for the daily means."""
    by_day: dict[date, list[float]] = {}
    for row in rows:
        by_day.setdefault(_local_date(row.timestamp, tz), []).append(
            row.predicted - row.actual
        )
    return error_stats(
        [
            (
                len(errors),
                sum(errors),
                sum(map(abs, errors)),
                sum(e * e for e in errors),
            )
            for errors in by_day.values()
        ]
    )


def bucket_stats(data: LiveData) -> dict[str, ErrorStats]:
    """Return the stats per lead-time bucket over every retained day."""
    stats: dict[str, ErrorStats] = {}
    for bucket in _ordered_buckets(data):
        combined = error_stats(
            [sums for (_, key), sums in data.lead_time.items() if key == bucket]
        )
        if combined is not None:
            stats[bucket] = combined
    return stats


def weekly_stats(data: LiveData) -> dict[date, dict[str, ErrorStats]]:
    """Return the stats per ISO week (keyed by its Monday) and bucket."""
    grouped: dict[date, dict[str, list[ErrorSums]]] = {}
    for (day, bucket), sums in data.lead_time.items():
        monday = day - timedelta(days=day.weekday())
        grouped.setdefault(monday, {}).setdefault(bucket, []).append(sums)
    weekly: dict[date, dict[str, ErrorStats]] = {}
    for monday in sorted(grouped):
        weekly[monday] = {}
        for bucket, daily in grouped[monday].items():
            combined = error_stats(daily)
            if combined is not None:
                weekly[monday][bucket] = combined
    return weekly


def training_times(log_text: str) -> list[float]:
    """Return every training duration (s) in Home Assistant log text."""
    return [float(match) for match in TRAINING_LOG.findall(log_text)]


def _ordered_buckets(data: LiveData) -> list[str]:
    """Return the export's buckets, known ones in lead-time order first."""
    present = {bucket for _, bucket in data.lead_time}
    return [b for b in _BUCKET_ORDER if b in present] + sorted(
        present - set(_BUCKET_ORDER)
    )


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    """Render a markdown table, first column left-aligned, the rest right."""
    align = [":--"] + ["--:"] * (len(header) - 1)
    return [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(align) + " |",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


def _optional_float(value: str | None) -> float | None:
    """Parse a meta value as a finite float; None if missing or invalid."""
    try:
        number = float(value) if value is not None else None
    except ValueError:
        return None
    return number if number is not None and math.isfinite(number) else None


def _coverage_lines(data: LiveData) -> list[str]:
    """Describe the period the export covers."""
    lines = ["## Coverage", ""]
    days = sorted({day for day, _ in data.lead_time})
    if days:
        lines.append(
            f"- Lead-time accuracy: {days[0]} to {days[-1]} (local slot dates), "
            f"{len(days)} day(s) with samples; kept for {LEAD_TIME_WINDOW_DAYS} days."
        )
        if len(days) < MIN_REPORT_DAYS:
            lines.append(
                f"- **Only {len(days)} day(s)**: record live accuracy from "
                f"{MIN_REPORT_DAYS}+ days of one integration version."
            )
    else:
        lines.append("- Lead-time accuracy: no rows.")
    if data.evaluation:
        lines.append(
            f"- Evaluation: {data.evaluation[0].timestamp} to "
            f"{data.evaluation[-1].timestamp} (UTC slots), {len(data.evaluation)} "
            f"slot(s); kept for {EVALUATION_KEEP_DAYS} days."
        )
    else:
        lines.append("- Evaluation: no rows.")
    trained_at = data.meta.get("holdout_trained_at")
    if trained_at:
        lines.append(f"- Latest training: {trained_at}.")
    return lines


def _stats_row(label: str, stats: ErrorStats, units: Units) -> list[str]:
    """Render one row of a per-bucket table."""
    return [
        label,
        str(stats.samples),
        str(stats.days),
        units.fmt(stats.mae),
        units.fmt(stats.rmse),
        units.fmt(stats.bias),
        units.fmt(stats.daily_mae),
        units.fmt(stats.daily_rmse),
    ]


_STATS_HEADER = (
    "Samples",
    "Days",
    "MAE",
    "RMSE",
    "Bias",
    "Daily MAE",
    "Daily RMSE",
)


def _accuracy_lines(data: LiveData, units: Units, tz: tzinfo) -> list[str]:
    """Render the per-bucket, per-week and evaluation tables."""
    lines = ["## Lead-time accuracy", ""]
    per_bucket = bucket_stats(data)
    if per_bucket:
        lines += _table(
            ["Bucket", *_STATS_HEADER],
            [_stats_row(b, s, units) for b, s in per_bucket.items()],
        )
        lines += [
            "",
            "MAE/RMSE/Bias pool every sample (as the accuracy sensors do; "
            "bias = mean of predicted - actual). Daily MAE/RMSE are the mean "
            "daily MAE and the root of the mean daily MSE (the backtest's method).",
        ]
    else:
        lines.append("No lead-time accuracy samples.")
    weekly = weekly_stats(data)
    if len(weekly) > 1:
        buckets = _ordered_buckets(data)
        lines += ["", "### Per week (MAE, samples)", ""]
        lines += _table(
            ["Week of", *buckets],
            [
                [
                    str(monday),
                    *(
                        f"{units.fmt(week[b].mae)} ({week[b].samples})"
                        if b in week
                        else "–"
                        for b in buckets
                    ),
                ]
                for monday, week in weekly.items()
            ],
        )
    lines += ["", "## Evaluation (prediction closest to 24 h ahead)", ""]
    evaluation = evaluation_stats(data.evaluation, tz)
    if evaluation is None:
        lines.append("No evaluation rows.")
    else:
        mean_lead = statistics.fmean(row.lead_hours for row in data.evaluation)
        lines += _table(
            ["Table", *_STATS_HEADER],
            [_stats_row("evaluation", evaluation, units)],
        )
        lines += ["", f"Mean lead time: {mean_lead:.1f} h."]
    return lines


def _training_lines(
    data: LiveData, units: Units, times: list[float] | None
) -> list[str]:
    """Render the holdout metrics, obsolete HPO keys and training times."""
    lines = ["## Training", ""]
    mae = _optional_float(data.meta.get("holdout_mae"))
    rmse = _optional_float(data.meta.get("holdout_rmse"))
    if mae is None or rmse is None:
        lines.append("- Holdout: none stored (no successful training).")
    else:
        lines.append(
            f"- Holdout (newest 20 % of the window): MAE {units.fmt(mae)}, "
            f"RMSE {units.fmt(rmse)}."
        )
    if "training_samples" in data.meta:
        lines.append(f"- Training samples: {data.meta['training_samples']}.")
    hpo = {key: data.meta[key] for key in OBSOLETE_HPO_META_KEYS if key in data.meta}
    best = _optional_float(hpo.get("hpo_best_mae"))
    if best is not None:
        hpo["hpo_best_mae"] = units.fmt(best)
    if hpo:
        values = ", ".join(f"`{key}` = {value}" for key, value in hpo.items())
        lines.append(
            f"- HPO parameters (obsolete since #92, the export predates it): {values}."
        )
    if times is not None:
        if times:
            lines.append(
                f"- Training time from the log ({len(times)} trainings): median "
                f"{statistics.median(times):.1f} s, min {min(times):.1f} s, "
                f"max {max(times):.1f} s."
            )
        else:
            lines.append("- Training time: no `ML model trained in … s` log lines.")
    return lines


def region_tz(region: str) -> ZoneInfo:
    """Return a region's time zone."""
    return ZoneInfo(str(REGIONS[region]["tz"]))


def format_report(
    data: LiveData,
    units: Units,
    region: str = "DK1",
    times: list[float] | None = None,
    since: date | None = None,
) -> str:
    """Render the live-accuracy report as markdown."""
    tz = region_tz(region)
    lines = [
        f"# Live accuracy: {data.path.name} ({region})",
        "",
        f"Prices in {units.label}."
        + (f" Slots from {since} on only." if since is not None else ""),
        "",
        *_coverage_lines(data),
        "",
        *_accuracy_lines(data, units, tz),
        "",
        *_training_lines(data, units, times),
    ]
    return "\n".join(lines) + "\n"


def units_from_args(args: argparse.Namespace) -> Units:
    """Return the report's units: the model's, or EUR ct/kWh."""
    if args.eur_per_unit is not None:
        if args.eur_per_unit <= 0:
            raise SystemExit("--eur-per-unit must be positive")
        return Units(args.eur_per_unit * 100, "EUR ct/kWh", 2)
    if args.currency is not None:
        per_eur = 1.0 if args.currency == "EUR" else PEGGED_EUR_RATES[args.currency]
        return Units(
            100 / per_eur,
            f"EUR ct/kWh (1 EUR = {per_eur:g} {args.currency})",
            2,
        )
    return Units()


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="python -m scripts.live_report",
        description="Live forecast accuracy from a learning-database export.",
    )
    parser.add_argument("db", type=Path, help="exported learning database (read-only)")
    parser.add_argument(
        "--region",
        default="DK1",
        choices=sorted(REGIONS),
        help="the export's region: its time zone groups the evaluation by day",
    )
    parser.add_argument(
        "--since",
        type=date.fromisoformat,
        help="only slots from this local date on, YYYY-MM-DD (e.g. the day "
        "after updating the instance)",
    )
    unit = parser.add_mutually_exclusive_group()
    unit.add_argument(
        "--eur-per-unit",
        type=float,
        help="EUR per unit of the model's currency (e.g. 0.134 for DKK): "
        "report in EUR ct/kWh",
    )
    unit.add_argument(
        "--currency",
        choices=["EUR", *sorted(PEGGED_EUR_RATES)],
        help="the model's currency, converted at a fixed rate (DKK: ERM II "
        "central rate): report in EUR ct/kWh; other currencies need --eur-per-unit",
    )
    parser.add_argument(
        "--log",
        type=Path,
        help="Home Assistant log: summarize the 'ML model trained in … s' lines",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Print the live-accuracy report of an export."""
    args = parse_args(argv)
    units = units_from_args(args)
    try:
        data = load_live_data(args.db, args.since, region_tz(args.region))
    except FileNotFoundError as err:
        raise SystemExit(f"No such export: {err}") from err
    times = None
    if args.log is not None:
        times = training_times(args.log.read_text(encoding="utf-8", errors="replace"))
    print(format_report(data, units, args.region, times, args.since), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
