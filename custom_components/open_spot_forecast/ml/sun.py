"""Sun position features for a bidding zone (#25).

Solar production follows the sun, not the clock: it starts at sunrise, peaks
around solar noon and ends at sunset, at times that move by hours over the
year. For every slot these features describe the sun at the zone's centre
(the mean of the region's ``WEATHER_POINTS``) at the slot's midpoint: its
elevation and azimuth in degrees, and the seconds since sunrise and since
sunset of the slot's local day (negative before them).

They are computed with ``astral``, the library behind Home Assistant's
``sun.sun`` entity, so they need no new dependency. ``sun.sun`` itself cannot
supply them: it only holds the sun's current position and next events at the
home location, while training needs every slot of the past window and
prediction the slots of the next week. A sunrise or sunset that does not
happen on a day (polar day or night) is None, NaN for the model.
"""

from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from functools import lru_cache
from statistics import fmean

from astral import Observer
from astral.sun import azimuth, elevation, sunrise, sunset

from ..const import WEATHER_POINTS

type ZoneCentre = tuple[float, float]

SUN_FEATURES: tuple[str, ...] = (
    "sun_elevation",
    "sun_azimuth",
    "since_sunrise",
    "since_sunset",
)


@lru_cache(maxsize=32)
def zone_centre(region: str | None) -> ZoneCentre | None:
    """Return the mean (latitude, longitude) of a region's weather points.

    None for no region or a region without points, whose sun features are
    then unknown.
    """
    points = WEATHER_POINTS.get(region) if region else None
    if not points:
        return None
    return fmean(lat for lat, _ in points), fmean(lon for _, lon in points)


def _event(
    event: Callable[..., datetime], observer: Observer, day: date
) -> datetime | None:
    """Return a sun event on ``day`` in UTC, or None if it does not happen."""
    try:
        return event(observer, day)
    except ValueError:
        return None


@lru_cache(maxsize=1024)
def _sun_events(day: date, centre: ZoneCentre) -> tuple[datetime | None, ...]:
    """Return (sunrise, sunset) on ``day`` at ``centre``."""
    observer = Observer(*centre)
    return _event(sunrise, observer, day), _event(sunset, observer, day)


# About 180 days of slots plus the forecast week, so a retrain on the same
# window reuses every row's sun position
@lru_cache(maxsize=32768)
def _sun_position(epoch: float, centre: ZoneCentre) -> tuple[float, float]:
    """Return the sun's (elevation, azimuth) in degrees at a UTC epoch."""
    moment = datetime.fromtimestamp(epoch, UTC)
    observer = Observer(*centre)
    return float(elevation(observer, moment)), float(azimuth(observer, moment))


def sun_features(
    start: datetime, centre: ZoneCentre | None, interval_minutes: int = 15
) -> dict[str, float | None]:
    """Return the sun features of the slot beginning at ``start``.

    Args:
        start: Timezone-aware slot start in the price region's local time;
            its local date selects the day's sunrise and sunset.
        centre: The zone's centre (``zone_centre``), or None if unknown.
        interval_minutes: Slot length; the sun is taken at the slot's middle.

    Returns:
        ``SUN_FEATURES``: elevation and azimuth (degrees), and the seconds
        from the day's sunrise and sunset to the slot's middle. All None
        without a centre; a sunrise or sunset that does not happen is None.
    """
    if centre is None:
        return dict.fromkeys(SUN_FEATURES)
    middle = start.astimezone(UTC) + timedelta(minutes=interval_minutes / 2)
    sun_elevation, sun_azimuth = _sun_position(middle.timestamp(), centre)
    rise, set_ = _sun_events(start.date(), centre)
    return {
        "sun_elevation": sun_elevation,
        "sun_azimuth": sun_azimuth,
        "since_sunrise": (middle - rise).total_seconds() if rise else None,
        "since_sunset": (middle - set_).total_seconds() if set_ else None,
    }
