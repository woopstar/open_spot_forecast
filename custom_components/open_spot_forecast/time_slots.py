"""Price-slot time arithmetic shared by the component and the ML layer.

Day-ahead prices come in 15-minute slots. Slot boundaries are computed on
the UTC timeline: every UTC offset in use is a whole multiple of 15 minutes,
so a 15-minute boundary in UTC is one in local time too, and a DST change
cannot shift the result.
"""

from datetime import UTC, datetime, timedelta

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
