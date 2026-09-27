"""ENTSO-E's week-ahead load forecast as a 15-minute curve (#30).

Nordpool's consumption prognosis only covers today and tomorrow, so the
model has no demand input for days 3-7 of the forecast. ENTSO-E's week-ahead
total load forecast (``documentType=A65``, ``processType=A31``) gives, for
each day of the coming week, the minimum (``businessType`` A60) and maximum
(A61) load of the bidding zone. Like EpexPredictor (BSD-3-Clause,
reimplemented), the two daily values become a curve: the minimum at 03:00,
the maximum at 11:30 and 19:00 and ``(3 × max + min) / 4`` at 14:30, local
time, joined by a cubic spline and sampled every 15 minutes.

``EntsoeLoadSource`` keeps those samples in ``entsoe_load`` (MW per UTC
slot) through the shared ``TimeSeriesSource``: the training window's days
once (the week-ahead forecasts ENTSO-E published for them), and from
yesterday on at every forecast run, as ENTSO-E revises them. It needs an
ENTSO-E security token.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from datetime import UTC, date, datetime, time, timedelta, tzinfo
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

import numpy as np

from homeassistant.core import HomeAssistant

from ..const import REGIONS
from ..ml.series_storage import ENTSOE_LOAD
from ..time_series import TimeRange, day_chunks
from ..time_slots import UTC_KEY_FORMAT
from .entsoe import (
    async_entsoe_get,
    child_text,
    children,
    entsoe_period,
    period_bounds,
    time_series,
)
from .time_series_source import TimeSeriesSource

if TYPE_CHECKING:
    from ..ml.storage import LearningStorage

_LOGGER = logging.getLogger(__name__)

_SLOT_SECONDS = 15 * 60
_DAY = timedelta(days=1)
# One request covers at most this many local days
_MAX_REQUEST_DAYS = 31
_MINIMUM, _MAXIMUM = "A60", "A61"


def parse_entsoe_load(text: str, tz: tzinfo) -> dict[date, tuple[float, float]]:
    """Return the (minimum, maximum) load forecast in MW per local day.

    A day needs both values. Points finer than a day are combined per local
    day (the lowest minimum, the highest maximum). An acknowledgement ("no
    matching data") has no days.

    Raises:
        ValueError: If the text is not XML.
    """
    extremes: dict[str, dict[date, float]] = {_MINIMUM: {}, _MAXIMUM: {}}
    for series in time_series(text):
        values = extremes.get(child_text(series, "businessType") or "")
        if values is None:
            continue
        pick = min if values is extremes[_MINIMUM] else max
        for period in children(series, "Period"):
            bounds = period_bounds(period)
            if bounds is None:
                continue
            start, _, step = bounds
            first_day = start.astimezone(tz).date()
            for point in children(period, "Point"):
                position = child_text(point, "position")
                quantity = child_text(point, "quantity")
                if not position or not quantity:
                    continue
                offset = (int(position) - 1) * step
                day = (
                    first_day + timedelta(days=offset.days)
                    if step >= _DAY
                    else (start + offset).astimezone(tz).date()
                )
                load = float(quantity)
                values[day] = pick(values.get(day, load), load)
    lows, highs = extremes[_MINIMUM], extremes[_MAXIMUM]
    return {day: (lows[day], highs[day]) for day in sorted(lows.keys() & highs.keys())}


def natural_cubic_spline(x: np.ndarray, y: np.ndarray, at: np.ndarray) -> np.ndarray:
    """Evaluate the natural cubic spline through ``(x, y)`` at ``at``.

    ``x`` must be strictly increasing with at least two points; ``at`` is
    expected within ``[x[0], x[-1]]``.
    """
    n = len(x)
    h = np.diff(x)
    # Second derivatives: zero at both ends, continuous slope inside
    system = np.zeros((n, n))
    rhs = np.zeros(n)
    system[0, 0] = system[-1, -1] = 1.0
    inner = np.arange(1, n - 1)
    system[inner, inner - 1] = h[:-1]
    system[inner, inner] = 2 * (h[:-1] + h[1:])
    system[inner, inner + 1] = h[1:]
    slopes = np.diff(y) / h
    rhs[inner] = 6 * np.diff(slopes)
    curvature = np.linalg.solve(system, rhs)
    index = np.clip(np.searchsorted(x, at, side="right") - 1, 0, n - 2)
    t = at - x[index]
    width = h[index]
    m0, m1 = curvature[index], curvature[index + 1]
    return np.asarray(
        y[index]
        + (slopes[index] - width * (2 * m0 + m1) / 6) * t
        + m0 / 2 * t**2
        + (m1 - m0) / (6 * width) * t**3
    )


def _anchors(low: float, high: float) -> Iterator[tuple[time, float]]:
    """Yield the day's curve anchors: local time and load (MW)."""
    yield time(3, 0), low
    yield time(11, 30), high
    yield time(14, 30), (3 * high + low) / 4
    yield time(19, 0), high


def _runs(days: list[date]) -> Iterator[list[date]]:
    """Yield runs of consecutive days (a gap splits the curve)."""
    run: list[date] = []
    for day in days:
        if run and day - run[-1] != _DAY:
            yield run
            run = []
        run.append(day)
    if run:
        yield run


def load_curve(
    days: dict[date, tuple[float, float]], tz: tzinfo
) -> list[dict[str, Any]]:
    """Return ``entsoe_load`` rows: the daily extremes as a 15-minute curve.

    The curve runs from the first day's 03:00 to the last day's 19:00 anchor
    of each run of consecutive days; slots outside it have no row.
    """
    rows: list[dict[str, Any]] = []
    for run in _runs(sorted(days)):
        points = [
            (datetime.combine(day, at, tz).timestamp(), load)
            for day in run
            for at, load in _anchors(*days[day])
        ]
        x = np.array([moment for moment, _ in points])
        y = np.array([load for _, load in points])
        first = -(-int(x[0]) // _SLOT_SECONDS) * _SLOT_SECONDS
        slots = np.arange(first, x[-1] + 1, _SLOT_SECONDS)
        # Hours from the first anchor keep the cubic terms well scaled
        loads = natural_cubic_spline((x - x[0]) / 3600, y, (slots - x[0]) / 3600)
        rows.extend(
            {
                "timestamp": datetime.fromtimestamp(int(slot), UTC).strftime(
                    UTC_KEY_FORMAT
                ),
                "load": round(float(load), 1),
            }
            for slot, load in zip(slots, loads, strict=True)
        )
    return rows


class EntsoeLoadSource(TimeSeriesSource):
    """ENTSO-E's week-ahead load forecast for one region's bidding zone."""

    spec = ENTSOE_LOAD
    # The week-ahead forecast is published once a day at most
    revalidate_after = timedelta(hours=3)

    def __init__(
        self,
        hass: HomeAssistant,
        storage: LearningStorage,
        region: str,
        api_key: str,
        horizon_cutoff: datetime | None = None,
    ) -> None:
        """Initialize the source for a region in ``REGIONS``.

        Args:
            hass: Home Assistant instance.
            storage: The learning database holding ``entsoe_load``.
            region: The region (bidding zone).
            api_key: ENTSO-E security token.
            horizon_cutoff: Never request or return data from this moment on.
        """
        super().__init__(hass, storage, horizon_cutoff)
        zone = REGIONS[region]
        self.eic = str(zone["entsoe"])
        self.tz = ZoneInfo(str(zone["tz"]))
        self._api_key = api_key

    def refresh_from(self, now: datetime) -> datetime | None:
        """Re-fetch from yesterday on: ENTSO-E revises the week ahead."""
        return now - _DAY

    def chunks(self, ranges: list[TimeRange]) -> list[TimeRange]:
        """Return requests of whole local days, at most ``_MAX_REQUEST_DAYS`` each."""
        return day_chunks(ranges, self.tz, _MAX_REQUEST_DAYS)

    async def _fetch(
        self, start: datetime, end: datetime
    ) -> list[dict[str, Any]] | None:
        """Fetch the days of ``[start, end)`` and one on each side.

        The neighbouring days' anchors shape the curve at the range's edges;
        only the range's own slots are returned.
        """
        text = await async_entsoe_get(
            self.hass,
            self._api_key,
            {
                "documentType": "A65",
                "processType": "A31",
                "outBiddingZone_Domain": self.eic,
            }
            | entsoe_period(start - _DAY, end + _DAY),
        )
        if text is None:
            return None
        try:
            days = parse_entsoe_load(text, self.tz)
        except ValueError as err:
            _LOGGER.warning("Unusable ENTSO-E load forecast: %s", err)
            return None
        return [
            row
            for row in load_curve(days, self.tz)
            if start <= datetime.fromisoformat(row["timestamp"]) < end
        ]
