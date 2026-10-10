"""Diagnostic sensor: the day-ahead prediction of the current slot (#113).

The forecast sensor only looks ahead, and the ``Forecast evaluation`` sensor
keeps its predicted-vs-actual series in attributes, which the recorder does
not turn into history. This sensor's state is the prediction that was made
about ``EVALUATION_LEAD_HOURS`` before the slot that is current now (the one
the evaluation keeps once the slot is scored), so Home Assistant records the
day-ahead forecast as a plain series: a history or statistics graph over
this sensor and the actual price shows predicted against actual for as long
as the recorder keeps them.

The price is converted like every exposed price (``PriceOutput``: the slot's
tariff, unit, surcharge, VAT), and is the hour's mean with ``hourly_average``.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from homeassistant.components.sensor import SensorEntity, SensorStateClass
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import EntityCategory
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.event import async_track_time_change
from homeassistant.util import dt as dt_util, slugify as util_slugify

from .attribution import ModelAttributionMixin
from .const import (
    DOMAIN,
    EVALUATION_LEAD_HOURS,
    UPDATE_SIGNAL,
    UPDATE_SIGNAL_FORECAST,
)
from .price_output import HOUR_MINUTES, PriceOutput
from .time_slots import SLOT_MINUTES, floor_to_slot


class DayAheadPredictionSensor(ModelAttributionMixin, SensorEntity):
    """The price predicted about a day ago for the slot (or hour) that is current."""

    _attr_has_entity_name = True
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_translation_key = "day_ahead_prediction"
    _attr_icon = "mdi:chart-timeline-variant"

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
        self.output = output
        self._attr_unique_id = util_slugify(
            f"{DOMAIN}_{entry.entry_id}_day_ahead_prediction"
        )
        self._attr_native_unit_of_measurement = output.unit(currency)
        self._attr_suggested_display_precision = output.precision
        self._attr_device_info = {"identifiers": {(DOMAIN, entry.entry_id)}}

    async def async_added_to_hass(self) -> None:
        """Refresh on every slot boundary and after each forecast or learning update."""
        for signal in (UPDATE_SIGNAL_FORECAST, UPDATE_SIGNAL):
            self.async_on_remove(
                async_dispatcher_connect(
                    self.hass, util_slugify(signal), self.async_write_ha_state
                )
            )
        # The state is another slot's prediction as soon as a slot starts
        self.async_on_remove(
            async_track_time_change(
                self.hass,
                self._handle_slot_start,
                minute=range(0, HOUR_MINUTES, SLOT_MINUTES),
                second=0,
            )
        )

    @callback
    def _handle_slot_start(self, _now: datetime) -> None:
        self.async_write_ha_state()

    @property
    def native_value(self) -> float | None:
        """Return the current slot's (or hour's) day-ahead prediction, None without one."""
        ml_predictor = self.api_data.get("ml_predictor")
        if ml_predictor is None:
            return None
        interval = self.output.interval_minutes
        first = floor_to_slot(dt_util.utcnow(), interval)
        predictions = [
            {"start": slot.isoformat(), "price": price}
            for slot in (
                first + timedelta(minutes=minutes)
                for minutes in range(0, interval, SLOT_MINUTES)
            )
            if (price := ml_predictor.day_ahead_prediction(slot)) is not None
        ]
        # One entry: the slot, or the mean of the hour's predicted slots
        forecast = self.output.forecast(predictions, self.api_data.get("tariffs"))
        return float(forecast[0]["price"]) if forecast else None

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return how far ahead the prediction was aimed to be made."""
        return {"lead_hours": EVALUATION_LEAD_HOURS}
