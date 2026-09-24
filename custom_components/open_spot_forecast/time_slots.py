"""Price-slot time arithmetic shared by the component and the ML layer.

Day-ahead prices come in 15-minute slots. Slot boundaries are computed on
the UTC timeline: every UTC offset in use is a whole multiple of 15 minutes,
so a 15-minute boundary in UTC is one in local time too, and a DST change
cannot shift the result.
"""

from collections.abc import Sequence
from datetime import UTC, date, datetime, time, timedelta, tzinfo

from homeassistant.util import dt as dt_util

SLOT_MINUTES = 15

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def floor_to_slot(moment: datetime, interval_minutes: int = SLOT_MINUTES) -> datetime:
    """Return the start of the slot that contains ``moment``.

    Args:
        moment: Timezone-aware datetime.
        interval_minutes: Slot length in minutes (a divisor of 60).

    Returns:
        The slot start, in the same time zone as ``moment``.
    """
    utc = moment.astimezone(UTC)
    utc -= (utc - _EPOCH) % timedelta(minutes=interval_minutes)
    return utc.astimezone(moment.tzinfo)


def ceil_to_slot(moment: datetime, interval_minutes: int = SLOT_MINUTES) -> datetime:
    """Return ``moment`` if it is a slot boundary, else the next boundary.

    Args:
        moment: Timezone-aware datetime.
        interval_minutes: Slot length in minutes (a divisor of 60).

    Returns:
        The boundary, in the same time zone as ``moment``.
    """
    utc = moment.astimezone(UTC)
    utc += (_EPOCH - utc) % timedelta(minutes=interval_minutes)
    return utc.astimezone(moment.tzinfo)


def first_prediction_slot(
    now: datetime,
    known_data_end_time: datetime | None = None,
    interval_minutes: int = SLOT_MINUTES,
) -> datetime:
    """Return the start of the first slot to predict.

    That is the slot containing ``now`` (the current slot), unless confirmed
    prices reach further: then it is the first slot boundary at or after
    ``known_data_end_time``, so no known slot is predicted and none is
    skipped.

    Args:
        now: Current time, timezone-aware.
        known_data_end_time: End of the last confirmed price slot, if any.
        interval_minutes: Slot length in minutes (a divisor of 60).

    Returns:
        Start of the first slot to predict, in UTC.
    """
    # Compare in UTC: two datetimes in the same zone compare by wall clock,
    # which is ambiguous in the repeated hour of a DST fall-back
    start = floor_to_slot(now.astimezone(UTC), interval_minutes)
    if known_data_end_time is not None:
        known_end = ceil_to_slot(known_data_end_time.astimezone(UTC), interval_minutes)
        start = max(start, known_end)
    return start


def local_midnight(day: date, tz: tzinfo | None = None) -> datetime:
    """Return the start of the local calendar ``day``.

    Args:
        day: Local calendar date.
        tz: Time zone; Home Assistant's time zone by default.

    Returns:
        Local midnight as a timezone-aware datetime in ``tz``.
    """
    return datetime.combine(day, time(), tzinfo=tz or dt_util.get_default_time_zone())


def slots_in_local_day(
    day: date, interval_minutes: int = SLOT_MINUTES, tz: tzinfo | None = None
) -> int:
    """Return the number of slots in a local calendar day.

    96 for 15-minute slots on a normal day; 92 on the spring-forward day and
    100 on the fall-back day, since those days last 23 and 25 hours.

    Args:
        day: Local calendar date.
        interval_minutes: Slot length in minutes (a divisor of 60).
        tz: Time zone; Home Assistant's time zone by default.

    Returns:
        The number of slots between the day's local midnight and the next.
    """
    # Subtract in UTC: aware datetimes sharing a tzinfo subtract by wall clock
    start = local_midnight(day, tz).astimezone(UTC)
    end = local_midnight(day + timedelta(days=1), tz).astimezone(UTC)
    return (end - start) // timedelta(minutes=interval_minutes)


def slot_start_in_day(
    day: date,
    index: int,
    tz: tzinfo | None = None,
    interval_minutes: int = SLOT_MINUTES,
) -> datetime:
    """Return the start of slot ``index`` of a local day, in local time.

    Steps on the UTC timeline from local midnight, so every slot after a DST
    change gets its real wall-clock time and offset: slot 8 of 2026-03-29 in
    Europe/Copenhagen is 03:00+02:00, and slots 8 and 12 of 2026-10-25 are
    both 02:00, at +02:00 and +01:00.

    Args:
        day: Local calendar date.
        index: Slot position from local midnight (0-91/95/99 for 15 minutes).
        tz: Time zone; Home Assistant's time zone by default.
        interval_minutes: Slot length in minutes (a divisor of 60).

    Returns:
        The slot start as a timezone-aware datetime in ``tz``.
    """
    midnight = local_midnight(day, tz)
    utc_start = midnight.astimezone(UTC) + index * timedelta(minutes=interval_minutes)
    return utc_start.astimezone(midnight.tzinfo)


def slot_index_in_day(
    moment: datetime,
    tz: tzinfo | None = None,
    interval_minutes: int = SLOT_MINUTES,
) -> int:
    """Return the position of the slot containing ``moment`` in its local day.

    The inverse of ``slot_start_in_day``: counted on the UTC timeline from
    local midnight, so on DST days it indexes a 92- or 100-slot day correctly
    instead of assuming ``hour * 4 + minute // 15``.

    Args:
        moment: Timezone-aware datetime.
        tz: Time zone of the day; Home Assistant's time zone by default.
        interval_minutes: Slot length in minutes (a divisor of 60).

    Returns:
        Slots between local midnight of ``moment``'s local date and its slot.
    """
    zone = tz or dt_util.get_default_time_zone()
    midnight = local_midnight(moment.astimezone(zone).date(), zone)
    elapsed = floor_to_slot(moment, interval_minutes).astimezone(
        UTC
    ) - midnight.astimezone(UTC)
    return elapsed // timedelta(minutes=interval_minutes)


def tomorrow_prices_complete(
    prices: Sequence[float], now: datetime | None = None
) -> bool:
    """Return True if ``prices`` has a price for every slot of tomorrow.

    Tomorrow is the next local calendar day, so a DST-change day needs 92 or
    100 prices instead of 96. A partial publication is not complete.

    Args:
        prices: Tomorrow's 15-minute prices.
        now: Current time (defaults to now), used to find tomorrow's date.

    Returns:
        True if there are at least as many prices as tomorrow has slots.
    """
    today = dt_util.as_local(now or dt_util.utcnow()).date()
    return len(prices) >= slots_in_local_day(today + timedelta(days=1))
