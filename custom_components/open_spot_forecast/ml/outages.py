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
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

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
        self._versions: dict[str, dict[int, _Version]] = {}
        self._periods: dict[tuple[str, int], list[OutagePeriod]] = {}
        self._known: dict[int, list[OutagePeriod]] = {}
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
            self._versions.setdefault(message_id, {})[version] = _Version(
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
            self._periods.setdefault((message_id, version), []).append(
                OutagePeriod(str(kind), start, stop, unavailable)
            )

    def known_at(self, origin: datetime) -> list[OutagePeriod]:
        """Return the periods in force as published at or before ``origin``.

        Per message the latest version published by then; none if that
        version is dismissed.
        """
        at = int(origin.timestamp())
        cached = self._known.get(at)
        if cached is not None:
            return cached
        periods: list[OutagePeriod] = []
        for message_id, versions in self._versions.items():
            latest = max(
                (v for v in versions.values() if v.published <= at),
                key=lambda v: (v.published, v.version),
                default=None,
            )
            if latest is None or latest.dismissed:
                continue
            periods.extend(self._periods.get((message_id, latest.version), ()))
        self._known[at] = periods
        return periods

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
        sums = dict.fromkeys(OUTAGE_FEATURES.values(), 0.0)
        for period in self.known_at(origin):
            if period.start <= at < period.stop:
                sums[OUTAGE_FEATURES[period.kind]] += period.unavailable
        return sums
