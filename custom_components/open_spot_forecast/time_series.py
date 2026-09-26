"""Gap bookkeeping for time-series sources on a fixed UTC grid (#32).

A source (Nordpool prognoses, and later day-ahead prices and Open-Meteo
weather) stores one row per grid step. Before a request, the source works
out which parts of the wanted range are missing (``missing_ranges``), skips
what it already knows the upstream does not have yet (``SourceState``), and
fetches the rest in request-sized chunks (``split_range``).

All ranges are half-open ``[start, end)`` and in UTC.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Collection, Iterable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta, tzinfo

from .time_slots import ceil_to_slot, floor_to_slot, local_midnight, parse_utc

TimeRange = tuple[datetime, datetime]

# Stands in for "no end" when an open-ended hole takes part in range arithmetic
_FAR_FUTURE = datetime(9999, 1, 1, tzinfo=UTC)


def grid_range(start: datetime, end: datetime, step_minutes: int) -> TimeRange:
    """Widen ``[start, end)`` to the grid: floor the start, ceil the end (UTC)."""
    return (
        floor_to_slot(start, step_minutes).astimezone(UTC),
        ceil_to_slot(end, step_minutes).astimezone(UTC),
    )


def missing_ranges(
    known: Collection[datetime], start: datetime, end: datetime, step_minutes: int
) -> list[TimeRange]:
    """Return the runs of grid points in ``[start, end)`` that are not known.

    Args:
        known: Grid points that are stored (any time zone).
        start: Range start; widened down to the grid.
        end: Range end (exclusive); widened up to the grid.
        step_minutes: Grid step, a divisor of 60.

    Returns:
        Contiguous missing ranges in UTC, oldest first.
    """
    first, last = grid_range(start, end, step_minutes)
    step = timedelta(minutes=step_minutes)
    known_epochs = {int(moment.timestamp()) for moment in known}
    ranges: list[TimeRange] = []
    run_start: datetime | None = None
    moment = first
    while moment < last:
        if int(moment.timestamp()) in known_epochs:
            if run_start is not None:
                ranges.append((run_start, moment))
                run_start = None
        elif run_start is None:
            run_start = moment
        moment += step
    if run_start is not None:
        ranges.append((run_start, last))
    return ranges


def merge_ranges(ranges: Iterable[TimeRange]) -> list[TimeRange]:
    """Return the union of ``ranges`` as sorted, non-overlapping ranges."""
    merged: list[TimeRange] = []
    for start, end in sorted(r for r in ranges if r[0] < r[1]):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def subtract_ranges(
    ranges: Iterable[TimeRange], removed: Iterable[TimeRange]
) -> list[TimeRange]:
    """Return the parts of ``ranges`` not covered by ``removed``."""
    result = merge_ranges(ranges)
    for cut_start, cut_end in merge_ranges(removed):
        remaining: list[TimeRange] = []
        for start, end in result:
            if cut_end <= start or end <= cut_start:
                remaining.append((start, end))
                continue
            if start < cut_start:
                remaining.append((start, cut_start))
            if cut_end < end:
                remaining.append((cut_end, end))
        result = remaining
    return result


def intersect_ranges(
    ranges: Iterable[TimeRange], within: Iterable[TimeRange]
) -> list[TimeRange]:
    """Return the parts of ``ranges`` that lie inside ``within``."""
    merged = merge_ranges(ranges)
    return merge_ranges(
        (max(start, w_start), min(end, w_end))
        for start, end in merged
        for w_start, w_end in merge_ranges(within)
        if max(start, w_start) < min(end, w_end)
    )


def day_chunks(
    ranges: Iterable[TimeRange], tz: tzinfo, max_days: int
) -> list[TimeRange]:
    """Return requests of whole days in ``tz`` that cover ``ranges``.

    Every day touched by a range is requested; consecutive days are grouped
    into runs of at most ``max_days``. A day is midnight to midnight in
    ``tz`` (23 or 25 hours on a DST change), returned in UTC.
    """
    days: set[date] = set()
    for start, end in ranges:
        day = start.astimezone(tz).date()
        while local_midnight(day, tz) < end:
            days.add(day)
            day += timedelta(days=1)
    requests: list[TimeRange] = []
    run: list[date] = []

    def close() -> None:
        requests.append(
            (
                local_midnight(run[0], tz).astimezone(UTC),
                local_midnight(run[-1] + timedelta(days=1), tz).astimezone(UTC),
            )
        )

    for day in sorted(days):
        if run and (day - run[-1] > timedelta(days=1) or len(run) == max_days):
            close()
            run = []
        run.append(day)
    if run:
        close()
    return requests


def split_range(time_range: TimeRange, max_span: timedelta) -> list[TimeRange]:
    """Split a range into consecutive chunks no longer than ``max_span``."""
    start, end = time_range
    chunks: list[TimeRange] = []
    while start < end:
        chunk_end = min(start + max_span, end)
        chunks.append((start, chunk_end))
        start = chunk_end
    return chunks


@dataclass(slots=True)
class Hole:
    """A range the source did not have at its last request.

    ``end`` None means open-ended: the source had no data from ``start`` on
    (its known horizon), so later, longer requests skip it as well.
    """

    start: datetime
    end: datetime | None
    retry_at: datetime

    @property
    def range(self) -> TimeRange:
        """Return the hole as a range, an open end as the far future."""
        return (self.start, self.end or _FAR_FUTURE)


@dataclass(slots=True)
class SourceState:
    """What a source was missing at its last requests, and when to ask again."""

    holes: list[Hole] = field(default_factory=list)

    def pending(self, now: datetime) -> list[TimeRange]:
        """Return the ranges not to request before their retry time."""
        return [hole.range for hole in self.holes if now < hole.retry_at]

    @property
    def horizon(self) -> datetime | None:
        """Return where the source's data ended at its last request, if known."""
        return next((hole.start for hole in self.holes if hole.end is None), None)

    def record(
        self,
        attempted: Iterable[TimeRange],
        still_missing: Iterable[TimeRange],
        open_from: datetime | None,
        retry_at: Callable[[TimeRange], datetime],
    ) -> None:
        """Replace the holes in the attempted ranges by what is still missing.

        Args:
            attempted: Ranges requested successfully in this update.
            still_missing: Parts of them the source did not deliver.
            open_from: Start of a missing range that reaches the end of the
                requested future, recorded as the source's horizon (open-ended).
            retry_at: When a hole may be requested again.
        """
        requested = merge_ranges(attempted)
        kept: list[Hole] = []
        for hole in self.holes:
            if hole.end is None:
                # A retried horizon is replaced by what this update found
                if not intersect_ranges([hole.range], requested):
                    kept.append(hole)
                continue
            kept.extend(
                Hole(start, end, hole.retry_at)
                for start, end in subtract_ranges([hole.range], requested)
            )
        for start, end in intersect_ranges(still_missing, requested):
            open_ended = open_from is not None and start == open_from
            kept.append(
                Hole(start, None if open_ended else end, retry_at((start, end)))
            )
        self.holes = sorted(kept, key=lambda hole: hole.start)

    def prune(self, before: datetime) -> None:
        """Forget holes that end before ``before`` (outside retention)."""
        self.holes = [
            hole for hole in self.holes if hole.end is None or hole.end > before
        ]

    def to_json(self) -> str:
        """Serialise the state for the ``meta`` table."""
        return json.dumps(
            [
                [
                    hole.start.isoformat(),
                    hole.end.isoformat() if hole.end else None,
                    hole.retry_at.isoformat(),
                ]
                for hole in self.holes
            ]
        )

    @classmethod
    def from_json(cls, value: str | None) -> SourceState:
        """Read a state written by ``to_json``; anything unreadable is dropped."""
        try:
            items = json.loads(value) if value else []
        except ValueError:
            return cls()
        holes: list[Hole] = []
        for item in items if isinstance(items, list) else []:
            if not isinstance(item, list) or len(item) != 3:
                continue
            start, end, retry_at = (parse_utc(part) for part in item)
            if start is None or retry_at is None or (end is None and item[1]):
                continue
            holes.append(Hole(start, end, retry_at))
        return cls(holes)
