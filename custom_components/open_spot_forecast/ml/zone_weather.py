"""Zone weather: Open-Meteo sampling points aggregated per slot (#22).

A region's weather is sampled at a few fixed points across its bidding zone
(``WEATHER_POINTS``), every 15 minutes: wind speed at 80 m (turbine hub
height), temperature, global irradiance, sea-level pressure and humidity.
The model sees zone aggregates, the same for every region whatever its
number of points: the mean of each value, and the mean turbine power curve
of the points' wind (a mean of a curve, not the curve of the mean, so a
zone where some points are calm and some are windy is not averaged away).

Training and prediction both build these aggregates from the stored
``openmeteo_weather`` rows with ``ZoneWeatherIndex``; a slot without rows
has unknown (None) zone inputs.
"""

from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from typing import Any

from ..const import WEATHER_POINTS
from .features import floor_epoch, optional_float, utc_epoch, wind_power_curve

# The stored values of one point and slot (table columns)
ZONE_FIELDS = ("wind_80m", "temperature", "irradiance", "pressure", "humidity")


def point_key(point: tuple[float, float]) -> str:
    """Return the storage key of a sampling point, e.g. ``57.40,10.24``."""
    return f"{point[0]:.2f},{point[1]:.2f}"


def zone_points(region: str) -> list[str]:
    """Return the storage keys of a region's sampling points."""
    return [point_key(point) for point in WEATHER_POINTS.get(region, ())]


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def zone_aggregates(points: Iterable[dict[str, Any]]) -> dict[str, float | None]:
    """Return the ``SlotInputs`` zone fields from one slot's point rows.

    Each aggregate is over the points that have the value; None if none has.
    """
    values: dict[str, list[float]] = {name: [] for name in ZONE_FIELDS}
    for row in points:
        for name in ZONE_FIELDS:
            value = optional_float(row.get(name))
            if value is not None:
                values[name].append(value)
    winds = values["wind_80m"]
    return {
        "zone_wind": _mean(winds),
        "zone_wind_power": _mean([wind_power_curve(wind) for wind in winds]),
        "zone_temperature": _mean(values["temperature"]),
        "zone_irradiance": _mean(values["irradiance"]),
        "zone_pressure": _mean(values["pressure"]),
        "zone_humidity": _mean(values["humidity"]),
    }


class ZoneWeatherIndex:
    """Stored point rows, indexed by UTC slot, for per-slot zone aggregates."""

    def __init__(
        self, rows: Iterable[dict[str, Any]], points: Iterable[str] | None = None
    ) -> None:
        """Index ``openmeteo_weather`` rows.

        Args:
            rows: Rows with ``timestamp``, ``point`` and ``ZONE_FIELDS``.
            points: Only use these point keys (the region's current points);
                all points if None.
        """
        wanted = set(points) if points is not None else None
        self._slots: dict[int, list[dict[str, Any]]] = {}
        for row in rows:
            if wanted is not None and row.get("point") not in wanted:
                continue
            key = utc_epoch(row.get("timestamp"), UTC)
            if key is not None:
                self._slots.setdefault(key, []).append(row)

    def for_slot(self, start: datetime) -> dict[str, float | None]:
        """Return the zone aggregates of the slot beginning at ``start``."""
        return zone_aggregates(self._slots.get(floor_epoch(start, UTC), []))
