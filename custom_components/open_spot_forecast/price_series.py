"""Price series: alignment onto the 15-minute grid, and validation.

A day's prices are held as one value per 15-minute slot of the local day,
from local midnight: 96 values, or 92/100 on a DST-change day (see
``time_slots``). A slot without a price is ``None``: it is shown as missing
and is skipped by training and self-learning.
"""

import math
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, date, datetime, timedelta, tzinfo
from typing import Any

from .time_slots import (
    SLOT_MINUTES,
    floor_to_slot,
    local_midnight,
    slots_in_local_day,
)

# A price this close to 0 counts as zero (floats are never compared with ==)
_ZERO_PRICE_EPSILON = 1e-9

# Gaps of up to this many slots between two known prices are forward-filled
MAX_FILL_SLOTS = 4

# A sample never lasts longer than this (hourly prices)
_MAX_SAMPLE_DURATION = timedelta(minutes=60)

# (start, end or None, price) of one price from a source, timezone-aware
PriceSample = tuple[datetime, datetime | None, float]


def align_to_grid(
    samples: Iterable[PriceSample],
    day: date,
    tz: tzinfo | None = None,
    max_fill_slots: int = MAX_FILL_SLOTS,
) -> list[float | None]:
    """Place timestamped prices onto the 15-minute grid of a local day.

    Every sample fills the slots from its start to its end. A sample without
    an end lasts the series' resolution: the smallest gap between consecutive
    starts (15 minutes for quarter-hour prices, 60 for hourly ones, which
    expand to four slots), at most 60 minutes. A gap of at most
    ``max_fill_slots`` between two known prices takes the earlier price;
    longer gaps, and slots before the first or after the last price, stay
    ``None``. Samples outside the day and non-finite prices are ignored.

    Args:
        samples: ``(start, end or None, price)`` with timezone-aware times.
        day: Local calendar date of the grid.
        tz: Time zone of the day; Home Assistant's time zone by default.
        max_fill_slots: Longest gap, in slots, that is forward-filled.

    Returns:
        One value per slot of the day (92/96/100), or ``[]`` if no sample
        falls in the day.
    """
    slot = timedelta(minutes=SLOT_MINUTES)
    day_start = local_midnight(day, tz).astimezone(UTC)
    values: list[float | None] = [None] * slots_in_local_day(day, tz=tz)

    # Sort on the start only: equal starts must not compare their ends
    usable = sorted(
        (
            (start.astimezone(UTC), end.astimezone(UTC) if end else None, float(price))
            for start, end, price in samples
            if math.isfinite(price)
        ),
        key=lambda sample: sample[0],
    )
    starts = sorted({start for start, _, _ in usable})
    spacing = min((b - a for a, b in zip(starts, starts[1:])), default=slot)
    resolution = min(max(spacing, slot), _MAX_SAMPLE_DURATION)

    for start, end, price in usable:
        # From the slot holding the start up to (not incl.) the one holding
        # the end: at least one slot, even if the start is between boundaries
        first = (floor_to_slot(start) - day_start) // slot
        last = (floor_to_slot(end or start + resolution) - day_start) // slot
        for index in range(max(first, 0), min(max(last, first + 1), len(values))):
            values[index] = price

    known = [index for index, value in enumerate(values) if value is not None]
    if not known:
        return []
    for left, right in zip(known, known[1:]):
        if right - left - 1 <= max_fill_slots:
            for index in range(left + 1, right):
                values[index] = values[left]
    return values


def split_local_days(
    prices: Sequence[float | None], first_day: date, tz: tzinfo | None = None
) -> list[tuple[date, list[float | None]]]:
    """Split a series that starts at local midnight into its local days.

    Today's prices followed by tomorrow's are one list on the 15-minute
    grid; each day takes its own number of slots (92/96/100 on a DST-change
    day), so the next day starts at its own midnight.

    Args:
        prices: One value per 15-minute slot from local midnight of
            ``first_day`` on, None for a missing slot.
        first_day: Local calendar date of the first slot.
        tz: Time zone of the days; Home Assistant's time zone by default.

    Returns:
        ``(day, prices)`` per local day, in order; the last day is shorter
        when the series ends inside it. ``[]`` for an empty series.
    """
    days: list[tuple[date, list[float | None]]] = []
    day = first_day
    start = 0
    while start < len(prices):
        count = slots_in_local_day(day, tz=tz)
        days.append((day, list(prices[start : start + count])))
        start += count
        day += timedelta(days=1)
    return days


def known_prices(prices: Sequence[float | None]) -> list[float]:
    """Return the prices of the slots that have one, in order."""
    return [price for price in prices if price is not None]


def same_prices(first: Sequence[float | None], second: Sequence[float | None]) -> bool:
    """Return True if two series have the same slots and (near-)equal prices."""
    return len(first) == len(second) and all(
        (a is None and b is None)
        or (a is not None and b is not None and abs(a - b) <= _ZERO_PRICE_EPSILON)
        for a, b in zip(first, second)
    )


def carry_over_day_prices(
    previous: Mapping[str, Any] | None, read: Mapping[str, Any], today: date
) -> dict[str, Any]:
    """Return a today/tomorrow price read with the slots only ``previous`` knows.

    Both hold one price list per local day (``today``, ``tomorrow``) and
    ``day``, the local date of the ``today`` list (``today`` if missing). A
    price that was read stays known when a later read lacks it: an
    unavailable or rejected sensor (``is_invalid_price_series``) reads as
    empty, and after midnight the previous read's ``tomorrow`` is the new
    day's ``today`` until the source has rotated its own lists. A price in
    ``read`` always wins, and nothing is kept of a day ``read`` does not cover.

    Args:
        previous: The data stored by the last read, or None.
        read: The new read (``read_stromligning_sensor()``-shaped).
        today: The local date of the read.

    Returns:
        A copy of ``read`` with ``day`` set and the missing slots filled in.
    """
    result = dict(read)
    result["day"] = day = read.get("day") or today
    if not previous:
        return result
    previous_day: date = previous.get("day") or today
    stored = {
        previous_day + timedelta(days=offset): key
        for offset, key in enumerate(("today", "tomorrow"))
    }
    for offset, key in enumerate(("today", "tomorrow")):
        source = stored.get(day + timedelta(days=offset))
        old = list(previous.get(source) or []) if source else []
        new = list(read.get(key) or [])
        if old and not new:
            result[key] = old
            result[f"raw_{key}"] = list(previous.get(f"raw_{source}") or [])
        elif len(new) == len(old):
            result[key] = [
                kept if price is None else price for price, kept in zip(new, old)
            ]
    return result


def is_invalid_price_series(prices: Sequence[float | None]) -> bool:
    """Return True if a day's prices must not be used.

    A failing price source (a sensor back from a restart, an API hiccup)
    often reports 0 for every slot. A day is invalid when every known price
    is zero, or when any price is not finite (NaN, inf). ``None`` marks a
    missing slot and is skipped. Some zero or negative prices are normal and
    keep a day valid. A series without any known price means "no data"
    rather than bad data, so it is not invalid.

    Args:
        prices: The day's prices, in any order.

    Returns:
        True if the known prices are all zero, or one is not finite.
    """
    known: list[float] = []
    for price in prices:
        if price is None:
            continue
        if not math.isfinite(price):
            return True
        known.append(price)
    return bool(known) and all(abs(price) <= _ZERO_PRICE_EPSILON for price in known)
