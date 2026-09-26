"""Per-slot training inputs from stored actuals and prognoses.

Training rows use the same ``build_feature_row`` as prediction rows; this
module supplies their ``SlotInputs``: the weather snapshot taken in the slot
(``weather_history``, stored every 15 minutes), the Nordpool prognoses for
the slot's hour (``nordpool_prognoses``) and the Open-Meteo zone weather for
the slot (``openmeteo_weather``, #22; aggregated by ``ZoneWeatherIndex``, as
at prediction). The tables are read once per fit and keyed by UTC epoch, so
a DST change or a mixed timestamp format cannot shift a row onto the wrong
slot.
"""

from collections.abc import Iterable
from datetime import datetime, tzinfo
from typing import Any

from .features import (
    HOUR_SECONDS,
    SlotInputs,
    floor_epoch,
    hour_epoch,
    optional_float,
    utc_epoch,
)
from .zone_weather import ZoneWeatherIndex


class TrainingInputs:
    """Stored weather snapshots and Nordpool prognoses, indexed by slot."""

    def __init__(
        self,
        weather_rows: Iterable[dict[str, Any]],
        nordpool_rows: Iterable[dict[str, Any]],
        tz: tzinfo,
        zone: ZoneWeatherIndex | None = None,
    ) -> None:
        """Index the rows; the first row in a slot (or hour) wins.

        Args:
            weather_rows: ``LearningStorage.load_weather_history()`` rows.
            nordpool_rows: ``LearningStorage.load_nordpool_history()`` rows.
            tz: Local time zone, for weather snapshots stored without an offset.
            zone: The stored Open-Meteo zone weather, if any.
        """
        self._tz = tz
        self._zone = zone
        self._weather: dict[int, dict[str, Any]] = {}
        for row in weather_rows:
            key = utc_epoch(row.get("timestamp"), tz)
            if key is not None:
                self._weather.setdefault(key, row)
        self._nordpool: dict[int, dict[str, Any]] = {}
        for row in nordpool_rows:
            key = hour_epoch(row.get("timestamp"), tz)
            if key is not None:
                self._nordpool.setdefault(key, row)

    def for_slot(self, start: datetime) -> SlotInputs:
        """Return the stored inputs for the slot beginning at ``start``."""
        weather = self._weather.get(floor_epoch(start, self._tz), {})
        nordpool = self._nordpool.get(floor_epoch(start, self._tz, HOUR_SECONDS), {})
        return SlotInputs(
            temperature=optional_float(weather.get("temperature")),
            wind_speed=optional_float(weather.get("wind_speed")),
            wind_direction=optional_float(weather.get("wind_direction")),
            cloud_coverage=optional_float(weather.get("cloud_coverage")),
            humidity=optional_float(weather.get("humidity")),
            consumption=optional_float(nordpool.get("consumption")),
            solar_generation=optional_float(nordpool.get("solar")),
            wind_offshore=optional_float(nordpool.get("wind_offshore")),
            wind_onshore=optional_float(nordpool.get("wind_onshore")),
            **(self._zone.for_slot(start) if self._zone else {}),
        )
