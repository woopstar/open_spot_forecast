"""Outages from Nord Pool's urgent market messages as model inputs (#123).

A planned or unplanned outage of a large plant or an interconnector moves
the price more than anything the weather or the prognoses show, and it is
not periodic, so bias correction cannot learn it. Nord Pool's UMMs (urgent
market messages) announce them: per message a list of units with their
unavailable capacity in time periods. A message is revised in numbered
versions (each with its publication time) and can be dismissed.

``OutageIndex`` holds the stored message versions and their periods (the
rows of ``umm_messages`` and ``umm_periods``, see ``api/nordpool_umm.py``
and, for the zones publishing on the ENTSO-E platform, ``api/entsoe_outages.py``, #138)
and aggregates them per slot **as known at an origin**: only versions
published at or before the origin count, the latest of them per message,
and a dismissed message counts for nothing. Training rows use the slot's
day-ahead gate (``day_ahead_gate``: 12:00 CET the day before the slot's
local day, when the auction that set the price closed) and prediction
rows the forecast's own time, so neither phase sees a message the market
did not have.

A version is its message's latest from its publication until the next
version's, so every period is known over one span of origins. The index
keeps the periods as arrays and answers a slot from a step function of the
slot's UTC day as known at the origin, built once per day and origin from
the periods in force then (#167): a slot costs a bisection, however many
periods the archive holds.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from itertools import zip_longest
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

from ..time_slots import parse_utc
from .features import floor_epoch, optional_float

# UMM ``messageType``: unavailability of production units, of transmission
# units (interconnectors). Consumption (2), market information (4) and other
# unavailability (5) are not inputs
UMM_PRODUCTION = 1
UMM_TRANSMISSION = 3
# UMM ``eventStatus``: a dismissed message's outage does not take place
UMM_ACTIVE = 1
UMM_DISMISSED = 3
# ``kind`` of a stored period, and the feature it is summed into
KIND_PRODUCTION = "production"
KIND_TRANSMISSION = "transmission"
OUTAGE_FEATURES: dict[str, str] = {
    KIND_PRODUCTION: "unavailable_production",
    KIND_TRANSMISSION: "unavailable_transmission",
}
_COLUMNS = {kind: column for column, kind in enumerate(OUTAGE_FEATURES)}
# The step functions cover one UTC day each: few breakpoints, however long
# the archive
_DAY_SECONDS = 86400
# ``known_until`` of a version no later one replaces
_FOREVER = int(np.iinfo(np.int64).max)

# The single day-ahead coupling closes at 12:00 CET for every delivery area
_MARKET_TZ = ZoneInfo("CET")
_GATE_CLOSURE = time(12, 0)


def day_ahead_gate(day: date) -> datetime:
    """Return when the day-ahead auction for local ``day`` closed (UTC).

    12:00 CET/CEST on the day before: the prices of ``day`` were set from
    what was published by then.
    """
    return datetime.combine(day - timedelta(days=1), _GATE_CLOSURE, _MARKET_TZ)


@dataclass(frozen=True, slots=True)
class OutagePeriod:
    """One unit's unavailable capacity (MW) over ``[start, stop)`` (UTC epochs)."""

    kind: str
    start: int
    stop: int
    unavailable: float


@dataclass(frozen=True, slots=True)
class _Version:
    """A message version: when it was published and whether it is dismissed."""

    published: int
    version: int
    dismissed: bool


@dataclass(frozen=True, slots=True)
class _Profile:
    """The unavailable capacity over one UTC day, as known at an origin.

    ``levels[i]`` (one value per ``OUTAGE_FEATURES`` entry) holds from
    ``breakpoints[i]`` until the next breakpoint, the last until the end of
    the day; the first breakpoint is the day's start.
    """

    breakpoints: list[int]
    levels: list[tuple[float, ...]]


def _epoch(value: Any) -> int | None:
    moment = parse_utc(value)
    return int(moment.timestamp()) if moment is not None else None


class OutageIndex:
    """Stored UMM rows, aggregated per slot as known at an origin."""

    def __init__(self, rows: Iterable[dict[str, Any]]) -> None:
        """Index message versions and their periods.

        Args:
            rows: One row per period, and one per version without a period
                (``kind`` None), with ``message_id``, ``version``,
                ``published``, ``status`` and, for a period, ``kind``,
                ``unit``, ``event_start``, ``event_stop`` and
                ``unavailable_mw``. A period given twice counts once.
        """
        versions: dict[str, dict[int, _Version]] = {}
        periods: dict[tuple[str, int], list[OutagePeriod]] = {}
        seen: set[tuple[Any, ...]] = set()
        for row in rows:
            message_id = row.get("message_id")
            published = _epoch(row.get("published"))
            try:
                version = int(row.get("version"))  # type: ignore[arg-type]
                status = int(row.get("status"))  # type: ignore[arg-type]
            except TypeError, ValueError:
                continue
            if not isinstance(message_id, str) or published is None:
                continue
            versions.setdefault(message_id, {})[version] = _Version(
                published, version, status == UMM_DISMISSED
            )
            kind = row.get("kind")
            start = _epoch(row.get("event_start"))
            stop = _epoch(row.get("event_stop"))
            unavailable = optional_float(row.get("unavailable_mw"))
            if (
                kind not in OUTAGE_FEATURES
                or start is None
                or stop is None
                or stop <= start
                or unavailable is None
                or unavailable <= 0
            ):
                continue
            key = (message_id, version, kind, row.get("unit"), start, stop)
            if key in seen:
                continue
            seen.add(key)
            periods.setdefault((message_id, version), []).append(
                OutagePeriod(str(kind), start, stop, unavailable)
            )
        # One row per period of a version that is not dismissed: the origins
        # it is known at (from its version's publication until the next
        # version's), its span and its feature column
        spans: list[tuple[int, int, int, int, int]] = []
        unavailable_mw: list[float] = []
        for message_id, numbered in versions.items():
            ordered = sorted(numbered.values(), key=lambda v: (v.published, v.version))
            for latest, following in zip_longest(ordered, ordered[1:]):
                if latest.dismissed:
                    continue
                known_until = following.published if following else _FOREVER
                for period in periods.get((message_id, latest.version), ()):
                    spans.append(
                        (
                            latest.published,
                            known_until,
                            period.start,
                            period.stop,
                            _COLUMNS[period.kind],
                        )
                    )
                    unavailable_mw.append(period.unavailable)
        table = np.array(spans, dtype=np.int64).reshape(-1, 5)
        (
            self._known_from,
            self._known_until,
            self._start,
            self._stop,
            self._column,
        ) = np.ascontiguousarray(table.T)
        self._unavailable = np.array(unavailable_mw, dtype=np.float64)
        self._profiles: dict[tuple[int, int], _Profile] = {}

    def _profile(self, day: int, known: int) -> _Profile:
        """Return the step function of UTC day ``day`` as known at ``known``.

        Per message the periods of the latest version published by then;
        none if that version is dismissed.
        """
        begin = day * _DAY_SECONDS
        end = begin + _DAY_SECONDS
        in_force = np.flatnonzero(
            (self._known_from <= known)
            & (known < self._known_until)
            & (self._start < end)
            & (begin < self._stop)
        )
        start = self._start[in_force]
        stop = self._stop[in_force]
        edges = np.concatenate(([begin], start, stop))
        breakpoints = np.unique(edges[(begin <= edges) & (edges < end)])
        levels = np.zeros((len(breakpoints), len(_COLUMNS)))
        if in_force.size:
            moments = breakpoints[:, None]
            active = (start <= moments) & (moments < stop)
            for column in _COLUMNS.values():
                counted = active & (self._column[in_force] == column)
                # Added period by period in the stored order, as a loop over
                # them would (``np.sum`` adds pairwise): a level is the exact
                # sum of the periods in force, not a running total that
                # drifts
                levels[:, column] = np.cumsum(
                    np.where(counted, self._unavailable[in_force], 0.0), axis=1
                )[:, -1]
        return _Profile(breakpoints.tolist(), [tuple(row) for row in levels.tolist()])

    def for_slot(self, start: datetime, origin: datetime) -> dict[str, float]:
        """Return the slot's unavailable MW per kind, as known at ``origin``.

        Args:
            start: Slot start (a naive one is UTC).
            origin: Only messages published at or before this moment count.

        Returns:
            The ``OUTAGE_FEATURES`` values: the sum of the unavailable
            capacity of every period containing the slot start, 0.0 when
            nothing is unavailable.
        """
        at = floor_epoch(start, UTC)
        key = (at // _DAY_SECONDS, int(origin.timestamp()))
        profile = self._profiles.get(key)
        if profile is None:
            profile = self._profiles[key] = self._profile(*key)
        level = profile.levels[bisect_right(profile.breakpoints, at) - 1]
        return dict(zip(OUTAGE_FEATURES.values(), level, strict=True))
