"""Live coverage of the p10/p90 bands from an export's ``interval_errors`` (#288).

The integration's band (``ml/intervals.py``, #280) spans the
``INTERVAL_QUANTILES`` of the raw model's past signed errors per (lead-time
bucket, local hour), over the last ``INTERVAL_WINDOW_DAYS`` scored days. The
backtest (``scripts/backtest_intervals.py``) measures it on one midnight
forecast per origin with archived weather; this replays it on the errors a
running instance stored, read by ``scripts/live_report.py`` from an export.

For every scored local day ``D``, the quantiles come from the rows of the
``INTERVAL_WINDOW_DAYS`` days before ``D`` (never ``D`` itself or later),
through the integration's own ``group_quantiles()``. A row of ``D`` is
inside its band when ``q_low <= error <= q_high`` of its ``interval_key()``:
with ``error = raw_price - actual`` that is ``raw_price - q_high <= actual <=
raw_price - q_low``, the band before it is widened to the served price.
``error > q_high`` is an actual price below the band, ``error < q_low`` one
above it. A row whose group had no quantiles counts as without a band.

The instance prunes the table to the last ``INTERVAL_WINDOW_DAYS`` days, so
an export's oldest days have fewer earlier days to replay than the live band
had, and its first ``MIN_INTERVAL_DAYS`` days have no band at all. Dev-only,
not shipped with the integration.
"""

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date, timedelta

from custom_components.open_spot_forecast.ml.intervals import (
    INTERVAL_WINDOW_DAYS,
    IntervalKey,
    group_quantiles,
    interval_key,
)

# One interval_errors row: (local slot date, bucket, slot 0-95, raw error)
IntervalRow = tuple[str, str, int, float]


@dataclass
class CoverageCounts:
    """Scored rows of one group and where their actual prices fell."""

    rows: int = 0
    inside: int = 0
    below: int = 0
    above: int = 0

    @property
    def banded(self) -> int:
        """Return the rows that had a band."""
        return self.inside + self.below + self.above

    def share(self, count: int) -> float | None:
        """Return ``count`` as a share of the banded rows; None without any."""
        return count / self.banded if self.banded else None

    def add(self, error: float, quantiles: tuple[float, float] | None) -> None:
        """Count one row's error against its group's quantiles (or none)."""
        self.rows += 1
        if quantiles is None:
            return
        q_low, q_high = quantiles
        if error > q_high:
            self.below += 1
        elif error < q_low:
            self.above += 1
        else:
            self.inside += 1


@dataclass
class IntervalCoverage:
    """The replayed band's coverage of an export's scored days."""

    # The scored local days, in order
    days: list[date] = field(default_factory=list)
    buckets: dict[str, CoverageCounts] = field(default_factory=dict)
    # (bucket, local hour) -> counts
    hours: dict[IntervalKey, CoverageCounts] = field(default_factory=dict)


def interval_coverage(
    rows: Iterable[IntervalRow],
    since: date | None = None,
    window_days: int = INTERVAL_WINDOW_DAYS,
) -> IntervalCoverage:
    """Replay the p10/p90 band on stored errors and count where prices fell.

    Args:
        rows: ``interval_errors`` rows.
        since: Score only days from this local date on; the bands still
            come from every earlier row, as the live band's did.
        window_days: Days before a scored day whose errors form its band.

    Returns:
        Counts per bucket and per (bucket, local hour), over the scored days.
    """
    by_day: dict[date, list[IntervalRow]] = {}
    for row in rows:
        by_day.setdefault(date.fromisoformat(row[0]), []).append(row)
    coverage = IntervalCoverage()
    for day in sorted(by_day):
        if since is not None and day < since:
            continue
        first = day - timedelta(days=window_days)
        quantiles = group_quantiles(
            row
            for earlier, earlier_rows in by_day.items()
            if first <= earlier < day
            for row in earlier_rows
        )
        coverage.days.append(day)
        for _, bucket, slot, error in by_day[day]:
            key = interval_key(bucket, slot)
            pair = quantiles.get(key)
            coverage.buckets.setdefault(bucket, CoverageCounts()).add(error, pair)
            coverage.hours.setdefault(key, CoverageCounts()).add(error, pair)
    return coverage
