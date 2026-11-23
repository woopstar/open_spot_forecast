"""Prediction interval coverage in the rolling backtest (#280, dev-only).

The integration's p10/p90 band (``ml/intervals.py``) spans the
``INTERVAL_QUANTILES`` of the forecast's past signed errors per lead-time
bucket and local hour, over the last ``INTERVAL_WINDOW_DAYS`` scored days. The backtest
replays it: a model's errors at every origin are kept per target slot, and
an origin's band comes from the errors of the target days before it (their
prices were known at the origin's midnight), through the integration's own
``group_quantiles()`` and ``interval_bounds()``. A forecast day ``d`` is
scored in the bucket of a ``d``-day lead (day 1 = ``day_1``).

Reported per forecast day: the share of actual prices inside the band
(target about 80 %), the share below and above it, the mean band width
(EUR ct/kWh) and the share of slots that had a band.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, tzinfo

import numpy as np

from custom_components.open_spot_forecast.const import LEAD_TIME_BUCKETS
from custom_components.open_spot_forecast.ml.intervals import (
    INTERVAL_QUANTILES,
    INTERVAL_WINDOW_DAYS,
    group_quantiles,
    interval_bounds,
    interval_key,
)


def horizon_bucket(day: int) -> str:
    """Return the lead-time bucket of forecast day ``day`` (1-based)."""
    return LEAD_TIME_BUCKETS[min(day, len(LEAD_TIME_BUCKETS)) - 1][0]


def wall_clock_slots(starts: np.ndarray, tz: tzinfo) -> np.ndarray:
    """Return each slot start's 15-minute slot of the local day by wall clock (0-95)."""
    slots = []
    for start in starts:
        local = datetime.fromtimestamp(int(start), tz)
        slots.append(local.hour * 4 + local.minute // 15)
    return np.asarray(slots, dtype=np.int64)


@dataclass
class _Origin:
    """One origin's forecast: per target slot its day, slot, horizon and prices."""

    origin: date
    days: list[date]
    slots: np.ndarray
    horizon: np.ndarray
    actual: np.ndarray
    predicted: np.ndarray


@dataclass
class IntervalCoverage:
    """One forecast day's interval scores."""

    inside: float
    below: float
    above: float
    width: float
    with_band: float


@dataclass
class IntervalScore:
    """Every origin's forecast of one model, scored as p10/p90 bands."""

    tz: tzinfo
    window_days: int = INTERVAL_WINDOW_DAYS
    _origins: list[_Origin] = field(default_factory=list)

    def add(
        self,
        origin: date,
        targets: np.ndarray,
        horizon: np.ndarray,
        actual: np.ndarray,
        predicted: np.ndarray,
    ) -> None:
        """Keep one origin's forecast (origins in increasing order)."""
        self._origins.append(
            _Origin(
                origin,
                [origin + timedelta(days=int(day) - 1) for day in horizon],
                wall_clock_slots(targets, self.tz),
                np.asarray(horizon),
                np.asarray(actual, dtype=float),
                np.asarray(predicted, dtype=float),
            )
        )

    def _errors_before(self, origin: date) -> list[tuple[str, str, int, float]]:
        """Return ``(date, bucket, slot, error)`` of the scored slots before ``origin``.

        The target days of the ``window_days`` before it.
        """
        first = origin - timedelta(days=self.window_days)
        rows: list[tuple[str, str, int, float]] = []
        end = bisect_left([item.origin for item in self._origins], origin)
        for item in self._origins[max(0, end - self.window_days - 8) : end]:
            for day, slot, day_index, actual, predicted in zip(
                item.days,
                item.slots,
                item.horizon,
                item.actual,
                item.predicted,
                strict=True,
            ):
                if first <= day < origin and not np.isnan(actual):
                    rows.append(
                        (
                            day.isoformat(),
                            horizon_bucket(int(day_index)),
                            int(slot),
                            float(predicted - actual),
                        )
                    )
        return rows

    def summary(self, horizon_days: int) -> list[IntervalCoverage | None]:
        """Return the interval scores per forecast day (None without any band)."""
        counts = np.zeros((horizon_days, 5))  # inside, below, above, width, slots
        totals = np.zeros(horizon_days)
        for item in self._origins:
            quantiles = group_quantiles(self._errors_before(item.origin))
            for slot, day_index, actual, predicted in zip(
                item.slots, item.horizon, item.actual, item.predicted, strict=True
            ):
                if np.isnan(actual) or not 1 <= day_index <= horizon_days:
                    continue
                row = int(day_index) - 1
                totals[row] += 1
                bounds = interval_bounds(
                    float(predicted),
                    float(predicted),
                    quantiles.get(
                        interval_key(horizon_bucket(int(day_index)), int(slot))
                    ),
                )
                if bounds is None:
                    continue
                low, high = bounds
                counts[row] += (
                    low <= actual <= high,
                    actual < low,
                    actual > high,
                    high - low,
                    1,
                )
        scores: list[IntervalCoverage | None] = []
        for row in range(horizon_days):
            banded = counts[row, 4]
            if not banded:
                scores.append(None)
                continue
            scores.append(
                IntervalCoverage(
                    inside=counts[row, 0] / banded,
                    below=counts[row, 1] / banded,
                    above=counts[row, 2] / banded,
                    width=counts[row, 3] / banded,
                    with_band=banded / totals[row],
                )
            )
        return scores


def _percent(share: float, decimals: int = 1) -> str:
    """Return a share as a percentage, e.g. ``80.4 %``."""
    return f"{share * 100:.{decimals}f} %"


def format_intervals(scores: dict[str, IntervalScore], horizon_days: int) -> str:
    """Render the p10/p90 coverage per model and forecast day as markdown."""
    lines = [
        "p10/p90 bands (errors per lead-time bucket and local hour, last "
        f"{INTERVAL_WINDOW_DAYS} scored days, quantiles "
        f"{INTERVAL_QUANTILES[0]:g}-{INTERVAL_QUANTILES[1]:g}): share of actual "
        "prices inside, "
        "below and above the band, mean width (EUR ct/kWh), share of slots "
        "with a band:",
        "",
        "| Model | Day | Inside | Below | Above | Width | With band |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, score in scores.items():
        for day, coverage in enumerate(score.summary(horizon_days), start=1):
            if coverage is None:
                lines.append(f"| {name} | {day} | n/a | n/a | n/a | n/a | 0 % |")
                continue
            lines.append(
                f"| {name} | {day} | {_percent(coverage.inside)} "
                f"| {_percent(coverage.below)} | {_percent(coverage.above)} "
                f"| {coverage.width:.2f} | {_percent(coverage.with_band, 0)} |"
            )
    return "\n".join(lines)
