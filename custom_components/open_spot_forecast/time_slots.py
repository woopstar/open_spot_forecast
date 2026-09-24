"""Price-slot time arithmetic shared by the component and the ML layer.

Day-ahead prices come in 15-minute slots. Slot boundaries are computed on
the UTC timeline: every UTC offset in use is a whole multiple of 15 minutes,
so a 15-minute boundary in UTC is one in local time too, and a DST change
cannot shift the result.
"""

from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta

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


def slots_in_local_day(day: date, interval_minutes: int = SLOT_MINUTES) -> int:
    """Return the number of slots in a local calendar day.

    96 for 15-minute slots on a normal day; 92 on the spring-forward day and
    100 on the fall-back day, since those days last 23 and 25 hours.

    Args:
        day: Local calendar date, in Home Assistant's time zone.
        interval_minutes: Slot length in minutes (a divisor of 60).

    Returns:
        The number of slots between the day's local midnight and the next.
    """
    # Subtract in UTC: aware datetimes sharing a tzinfo subtract by wall clock
    start = dt_util.start_of_local_day(day).astimezone(UTC)
    end = dt_util.start_of_local_day(day + timedelta(days=1)).astimezone(UTC)
    return (end - start) // timedelta(minutes=interval_minutes)


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
