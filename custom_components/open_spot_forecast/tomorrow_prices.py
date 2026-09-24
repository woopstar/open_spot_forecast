"""Re-check for tomorrow's day-ahead prices until the full day is published.

Day-ahead prices are published around 13:00 local time, sometimes later, and
the price sensor may update a while after that. From 13:00 local the prices
are re-read every ~5 minutes until tomorrow is complete (see
``tomorrow_prices_complete``); after that, not again until 13:00 the next
day. If tomorrow is still incomplete at 18:00, the checks stop for the day
with a warning.
"""

import logging
from collections.abc import Awaitable, Callable
from datetime import date, datetime, time, timedelta
from random import randint

from homeassistant.core import CALLBACK_TYPE, HomeAssistant, callback
from homeassistant.helpers.event import async_track_point_in_utc_time
from homeassistant.util import dt as dt_util

_LOGGER = logging.getLogger(__name__)

# Local time window in which tomorrow's prices are polled for
CHECK_START = time(13, 0)
CHECK_CUTOFF = time(18, 0)
CHECK_INTERVAL = timedelta(minutes=5)
# Up to this many seconds are added to each check time
MAX_JITTER_SECONDS = 30


def _local_moment(day: date, moment: time) -> datetime:
    """Return ``moment`` on the local ``day`` as a UTC datetime."""
    local = datetime.combine(day, moment, tzinfo=dt_util.get_default_time_zone())
    return dt_util.as_utc(local)


def next_tomorrow_check(now: datetime, tomorrow_complete: bool) -> datetime:
    """Return when to check for tomorrow's prices next (UTC, without jitter).

    Args:
        now: Current time, timezone-aware.
        tomorrow_complete: Whether tomorrow's prices are already complete.

    Returns:
        13:00 local today if that is still ahead; ``now`` + 5 minutes while
        tomorrow is incomplete before the 18:00 cutoff; otherwise 13:00 local
        on the next day.
    """
    today = dt_util.as_local(now).date()
    if not tomorrow_complete:
        start = _local_moment(today, CHECK_START)
        if now < start:
            return start
        if now < _local_moment(today, CHECK_CUTOFF):
            return dt_util.as_utc(now) + CHECK_INTERVAL
    return _local_moment(today + timedelta(days=1), CHECK_START)


class TomorrowPriceChecker:
    """Schedules tomorrow-price checks, one pending check at a time.

    ``check`` reads the prices and returns whether tomorrow is complete. Each
    check schedules the next one from its result, so polling stops by itself
    once tomorrow is complete and resumes at 13:00 the next day.
    """

    def __init__(
        self, hass: HomeAssistant, check: Callable[[], Awaitable[bool]]
    ) -> None:
        """Initialize the checker; nothing is scheduled until ``schedule``."""
        self._hass = hass
        self._check = check
        self._unsub: CALLBACK_TYPE | None = None

    @callback
    def schedule(self, tomorrow_complete: bool) -> None:
        """Replace any pending check with the next one due.

        Args:
            tomorrow_complete: Whether tomorrow's prices are already complete.
        """
        self.cancel()
        now = dt_util.utcnow()
        when = next_tomorrow_check(now, tomorrow_complete) + timedelta(
            seconds=randint(0, MAX_JITTER_SECONDS)
        )
        self._unsub = async_track_point_in_utc_time(self._hass, self._run, when)
        _LOGGER.debug("Next check for tomorrow's prices at %s", when)

    @callback
    def cancel(self) -> None:
        """Cancel the pending check, if any (called on unload)."""
        if self._unsub is not None:
            self._unsub()
            self._unsub = None

    async def _run(self, now: datetime) -> None:
        """Run one check, then schedule the next."""
        self._unsub = None
        complete = False
        try:
            complete = await self._check()
        except Exception:
            _LOGGER.exception("Error checking for tomorrow's prices")
        if not complete and now >= _local_moment(
            dt_util.as_local(now).date(), CHECK_CUTOFF
        ):
            _LOGGER.warning(
                "Tomorrow's prices are still incomplete after %s; "
                "checking again tomorrow from %s",
                CHECK_CUTOFF.strftime("%H:%M"),
                CHECK_START.strftime("%H:%M"),
            )
        self.schedule(complete)
