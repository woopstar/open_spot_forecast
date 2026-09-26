"""Open-Meteo weather at the region's sampling points (#22).

``OpenMeteoWeatherSource`` keeps the ``openmeteo_weather`` table (one row
per sampling point and UTC 15-minute slot) current through the shared
``TimeSeriesSource``. One request covers every point of the zone
(Open-Meteo takes comma-separated coordinate lists), and requests are
serialised: Open-Meteo dislikes concurrent calls. Forecasts are revised, so
the refresh window re-fetches from yesterday on; older slots keep the last
forecast fetched for them, which training uses.

Weather data by Open-Meteo.com, CC BY 4.0.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING, Any

from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from ..const import OPEN_METEO_API, WEATHER_POINTS
from ..ml.series_storage import OPENMETEO_WEATHER
from ..ml.zone_weather import point_key
from ..time_series import TimeRange, day_chunks
from .http import async_get
from .time_series_source import TimeSeriesSource

if TYPE_CHECKING:
    from ..ml.storage import LearningStorage

_LOGGER = logging.getLogger(__name__)

# Open-Meteo variable -> stored column
OPEN_METEO_VARIABLES: dict[str, str] = {
    "wind_speed_80m": "wind_80m",
    "temperature_2m": "temperature",
    "global_tilted_irradiance": "irradiance",
    "pressure_msl": "pressure",
    "relative_humidity_2m": "humidity",
}
# Wind in m/s, times in UTC; a horizontal panel (tilt 0) makes the tilted
# irradiance the global horizontal irradiance
OPEN_METEO_PARAMS: dict[str, str] = {
    "minutely_15": ",".join(OPEN_METEO_VARIABLES),
    "timezone": "UTC",
    "wind_speed_unit": "ms",
    "tilt": "0",
    "azimuth": "0",
}


def open_meteo_query(
    points: tuple[tuple[float, float], ...], first: date, last: date
) -> dict[str, str]:
    """Return the query for ``points`` from ``first`` to ``last`` (UTC days, inclusive)."""
    return OPEN_METEO_PARAMS | {
        "latitude": ",".join(f"{lat:.2f}" for lat, _ in points),
        "longitude": ",".join(f"{lon:.2f}" for _, lon in points),
        "start_date": first.isoformat(),
        "end_date": last.isoformat(),
    }


def parse_open_meteo(
    payload: Any, points: tuple[tuple[float, float], ...]
) -> list[dict[str, Any]]:
    """Return ``openmeteo_weather`` rows from an Open-Meteo response.

    The response lists one forecast per requested point, in request order
    (a single point comes as a bare object). Rows whose values are all
    missing are skipped.

    Raises:
        ValueError: If the response does not match the requested points.
    """
    forecasts = payload if isinstance(payload, list) else [payload]
    if len(forecasts) != len(points):
        raise ValueError(
            f"Open-Meteo returned {len(forecasts)} forecasts for {len(points)} points"
        )
    rows: list[dict[str, Any]] = []
    for point, forecast in zip(points, forecasts, strict=True):
        series = (forecast or {}).get("minutely_15") or {}
        times = series.get("time") or []
        columns = {
            column: series.get(variable) or []
            for variable, column in OPEN_METEO_VARIABLES.items()
        }
        key = point_key(point)
        for index, time_text in enumerate(times):
            values = {
                column: values[index] if index < len(values) else None
                for column, values in columns.items()
            }
            if all(value is None for value in values.values()):
                continue
            start = datetime.fromisoformat(time_text).replace(tzinfo=UTC)
            rows.append({"timestamp": start.isoformat(), "point": key} | values)
    return rows


class OpenMeteoWeatherSource(TimeSeriesSource):
    """Open-Meteo weather at one region's sampling points."""

    spec = OPENMETEO_WEATHER
    # Open-Meteo dislikes concurrent requests: one at a time, across sources
    _request_lock = asyncio.Lock()
    # Days per request (the archive's limit, #23)
    max_request_days = 90

    def __init__(
        self,
        hass: HomeAssistant,
        storage: LearningStorage,
        region: str,
        horizon_cutoff: datetime | None = None,
    ) -> None:
        """Initialize the source for a region in ``WEATHER_POINTS``."""
        super().__init__(hass, storage, horizon_cutoff)
        self.points = WEATHER_POINTS[region]

    def keys(self) -> list[str]:
        """Return the region's point keys: a slot needs a row for each."""
        return [point_key(point) for point in self.points]

    def refresh_from(self, now: datetime) -> datetime | None:
        """Re-fetch from yesterday on: recent forecasts are still revised."""
        return now - timedelta(days=1)

    def chunks(self, ranges: list[TimeRange]) -> list[TimeRange]:
        """Return requests of whole UTC days (Open-Meteo takes dates)."""
        return day_chunks(ranges, UTC, self.max_request_days)

    async def _fetch(
        self, start: datetime, end: datetime
    ) -> list[dict[str, Any]] | None:
        """Fetch every point for the UTC days ``[start, end)``."""
        first = start.astimezone(UTC).date()
        last = (end.astimezone(UTC) - timedelta(seconds=1)).date()
        async with self._request_lock:
            response = await async_get(
                async_get_clientsession(self.hass),
                OPEN_METEO_API,
                "Open-Meteo",
                params=open_meteo_query(self.points, first, last),
            )
        if response is None or response.status != 200:
            if response is not None:
                _LOGGER.warning("Open-Meteo returned %d", response.status)
            return None
        try:
            return parse_open_meteo(json.loads(response.text), self.points)
        except ValueError as err:
            _LOGGER.warning("Unusable Open-Meteo response: %s", err)
            return None
