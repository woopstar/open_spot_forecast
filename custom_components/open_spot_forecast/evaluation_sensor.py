"""Diagnostic sensor: the day-ahead forecast next to the actual prices (#36).

Once a slot's actual price is known its prediction leaves the forecast
sensor. This sensor keeps, for the last ``EVALUATION_WINDOW_HOURS``, the
prediction made closest to ``EVALUATION_LEAD_HOURS`` before each slot next
to the slot's actual price, as compact parallel arrays (see
``forecast_attributes``), so a dashboard can chart predicted against actual.
The predictions kept at the other ``EVALUATION_LEAD_TIMES`` (#113) are further
arrays aligned with them (``t12``, ``t48``; None where there is none).
Its state is the mean absolute error over that window. Prices are converted
like every exposed price (``PriceOutput``: unit, surcharge, VAT), with the
slot's tariff added to both (#107), so the error stays the spot price's.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from homeassistant.components.sensor import SensorEntity, SensorStateClass
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import EntityCategory
from homeassistant.core import HomeAssistant
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.util import dt as dt_util, slugify as util_slugify

from .attribution import ModelAttributionMixin
from .const import (
    DOMAIN,
    EVALUATION_LEAD_HOURS,
    EVALUATION_LEAD_TIMES,
    EVALUATION_WINDOW_HOURS,
    UPDATE_SIGNAL,
)
from .price_output import PriceOutput
from .time_slots import SLOT_MINUTES, parse_utc

# The other lead times a prediction is kept for (#113), by attribute name
EXTRA_LEAD_TIME_ARRAYS: dict[str, float] = {
    f"t{target:g}": target
    for target in EVALUATION_LEAD_TIMES
    if abs(target - EVALUATION_LEAD_HOURS) > 1e-9
}


class ForecastEvaluationSensor(ModelAttributionMixin, SensorEntity):
    """Mean absolute error of the day-ahead forecast, with the series behind it."""

    _attr_has_entity_name = True
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_translation_key = "forecast_evaluation"
    _attr_icon = "mdi:chart-bell-curve-cumulative"

    def __init__(
        self,
        hass: HomeAssistant,
        entry: ConfigEntry,
        api_data: dict[str, Any],
        currency: str,
        output: PriceOutput,
    ) -> None:
        """Initialize the sensor."""
        self.hass = hass
        self.api_data = api_data
        self.currency = currency
        self.output = output
        self._attr_unique_id = util_slugify(
            f"{DOMAIN}_{entry.entry_id}_forecast_evaluation"
        )
        self._attr_native_unit_of_measurement = output.unit(currency)
        self._attr_suggested_display_precision = output.precision
        self._attr_device_info = {"identifiers": {(DOMAIN, entry.entry_id)}}

    async def async_added_to_hass(self) -> None:
        """Refresh the state after every self-learning update."""
        self.async_on_remove(
            async_dispatcher_connect(
                self.hass, util_slugify(UPDATE_SIGNAL), self.async_write_ha_state
            )
        )

    def _window(self) -> list[tuple[datetime, dict[str, Any]]]:
        """Return the evaluated slots of the last window, prices converted."""
        ml_predictor = self.api_data.get("ml_predictor")
        if ml_predictor is None:
            return []
        since = dt_util.utcnow() - timedelta(hours=EVALUATION_WINDOW_HOURS)
        return [
            (start, row)
            for row in self.output.evaluation(
                ml_predictor.evaluation, self.api_data.get("tariffs")
            )
            if (start := parse_utc(row["start"])) is not None and start >= since
        ]

    @property
    def native_value(self) -> float | None:
        """Return the mean absolute error over the window, None without slots."""
        window = self._window()
        if not window:
            return None
        errors = [abs(row["predicted"] - row["actual"]) for _, row in window]
        return float(round(sum(errors) / len(errors), self.output.precision))

    def _extra_predictions(
        self, window: list[tuple[datetime, dict[str, Any]]]
    ) -> dict[str, list[float | None]]:
        """Return the other lead times' predictions, one per slot of the window."""
        ml_predictor = self.api_data.get("ml_predictor")
        snapshots = ml_predictor.evaluation_snapshots if ml_predictor else {}
        arrays: dict[str, list[float | None]] = {}
        for name, target in EXTRA_LEAD_TIME_ARRAYS.items():
            predicted = {
                row["start"]: row["predicted"]
                for row in self.output.evaluation(
                    snapshots.get(target, []), self.api_data.get("tariffs")
                )
            }
            arrays[name] = [predicted.get(row["start"]) for _, row in window]
        return arrays

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return the predicted and actual series as compact parallel arrays."""
        window = self._window()
        errors = [row["predicted"] - row["actual"] for _, row in window]
        return {
            "interval_minutes": SLOT_MINUTES,
            "unit": self.output.unit(self.currency),
            "lead_hours": EVALUATION_LEAD_HOURS,
            "window_hours": EVALUATION_WINDOW_HOURS,
            "samples": len(window),
            # Mean signed error: positive = the forecast was too high
            "bias": (
                float(round(sum(errors) / len(errors), self.output.precision))
                if errors
                else None
            ),
            "s": [int(start.timestamp()) for start, _ in window],
            "t": [row["predicted"] for _, row in window],
            "a": [row["actual"] for _, row in window],
            **self._extra_predictions(window),
        }
