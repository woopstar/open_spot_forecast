"""Public-holiday feature (#26).

On a public holiday offices and most industry are closed, so demand, and
with it the price, behaves like on a Sunday. ``holiday`` is 1 on Sundays and
public holidays and 0 on ordinary days, from the slot's local date. Where
public holidays differ within a bidding zone (Germany's states,
``HOLIDAY_SUBDIVISIONS``), it is the share of the zone's subdivisions with a
holiday that day. Christmas Eve and New Year's Eve are at least 0.5: in most
zones they are half days or de facto holidays.

Sundays count as holidays so the model can learn the holiday effect from the
Sundays in its training window: a 60-day window often has no weekday
holiday at all.

The calendars come from the ``holidays`` package, which Home Assistant's
``workday`` and ``holiday`` integrations use, and are built once per country
and year. Feature rows are only built in the executor (training and
prediction), never in the event loop.
"""

from datetime import date
from functools import lru_cache

import holidays

from ..const import HOLIDAY_SUBDIVISIONS, REGIONS

SUNDAY = 6
# (month, day) of the half days that count as at least half a holiday
HALF_DAYS = frozenset({(12, 24), (12, 31)})


@lru_cache(maxsize=32)
def _calendars(country: str, year: int) -> tuple[frozenset[date], ...]:
    """Return the public holidays of ``year``, one set per subdivision.

    One national set for a country without ``HOLIDAY_SUBDIVISIONS``.
    """
    subdivisions = HOLIDAY_SUBDIVISIONS.get(country) or (None,)
    return tuple(
        frozenset(holidays.country_holidays(country, subdiv=subdivision, years=year))
        for subdivision in subdivisions
    )


@lru_cache(maxsize=2048)
def public_holiday(day: date, region: str | None) -> float | None:
    """Return the ``holiday`` feature of a local date in a price region.

    Args:
        day: The slot's local calendar date.
        region: The price region (``REGIONS`` key).

    Returns:
        1.0 on Sundays; otherwise the share of the region's calendars with a
        public holiday on ``day`` (0 or 1 for a national calendar), at
        least 0.5 on Christmas Eve and New Year's Eve. None for no region or
        a region without a holiday calendar.
    """
    country = REGIONS.get(region, {}).get("holidays") if region else None
    if not country:
        return None
    if day.weekday() == SUNDAY:
        return 1.0
    calendars = _calendars(str(country), day.year)
    share = sum(day in calendar for calendar in calendars) / len(calendars)
    if (day.month, day.day) in HALF_DAYS:
        return max(share, 0.5)
    return share
