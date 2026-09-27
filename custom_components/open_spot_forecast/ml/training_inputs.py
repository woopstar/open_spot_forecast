"""Per-slot training inputs from stored forecasts and prognoses.

Training rows use the same ``build_feature_row`` as prediction rows; this
module supplies their ``SlotInputs``: the Open-Meteo zone weather for the
slot (``openmeteo_weather``: archived forecasts, #23, aggregated by
``ZoneWeatherIndex`` as at prediction), the Nordpool prognoses for the
slot's hour (``nordpool_prognoses``) and ENTSO-E's week-ahead load forecast
for the slot (``entsoe_load``, #30). All are forecasts, like the inputs
the model predicts from. The measured weather snapshots (``weather_history``)
are not training inputs any more (#23); they only score the local weather
forecast (confidence). The tables are read once per fit and keyed by UTC
epoch, so a DST change or a mixed timestamp format cannot shift a row onto
the wrong slot.
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
    slot_values,
)
from .zone_weather import ZoneWeatherIndex


class TrainingInputs:
    """Stored zone weather and Nordpool prognoses, indexed by slot."""

    def __init__(
        self,
        nordpool_rows: Iterable[dict[str, Any]],
        tz: tzinfo,
        zone: ZoneWeatherIndex | None = None,
        load_rows: Iterable[dict[str, Any]] = (),
    ) -> None:
        """Index the rows; the first row in an hour wins.

        Args:
            nordpool_rows: ``LearningStorage.load_nordpool_history()`` rows.
            tz: Local time zone, for timestamps stored without an offset.
            zone: The stored Open-Meteo zone weather, if any.
            load_rows: The stored ENTSO-E load forecast (``entsoe_load``, #30).
        """
        self._tz = tz
        self._zone = zone
        self._load = slot_values(load_rows, "load", tz)
        self._nordpool: dict[int, dict[str, Any]] = {}
        for row in nordpool_rows:
            key = hour_epoch(row.get("timestamp"), tz)
            if key is not None:
                self._nordpool.setdefault(key, row)

    def for_slot(self, start: datetime) -> SlotInputs:
        """Return the stored inputs for the slot beginning at ``start``."""
        nordpool = self._nordpool.get(floor_epoch(start, self._tz, HOUR_SECONDS), {})
        return SlotInputs(
            consumption=optional_float(nordpool.get("consumption")),
            solar_generation=optional_float(nordpool.get("solar")),
            wind_offshore=optional_float(nordpool.get("wind_offshore")),
            wind_onshore=optional_float(nordpool.get("wind_onshore")),
            load_forecast=self._load.get(floor_epoch(start, self._tz)),
            **(self._zone.for_slot(start) if self._zone else {}),
        )
