"""Binary sensor platform for Open Spot Forecast."""

import logging
from typing import Any

from homeassistant.components.binary_sensor import (
    BinarySensorDeviceClass,
    BinarySensorEntity,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.util import slugify as util_slugify

from .const import DOMAIN, UPDATE_SIGNAL

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Open Spot Forecast binary sensors."""
    api_data = hass.data[DOMAIN][entry.entry_id]

    binary_sensors = [
        TomorrowAvailableSensor(hass, entry, api_data),
        MLModelTrainedSensor(hass, entry, api_data),
    ]

    async_add_entities(binary_sensors, True)


class TomorrowAvailableSensor(BinarySensorEntity):
    """Binary sensor indicating if tomorrow's prices are available."""

    _attr_has_entity_name = True
    _attr_icon = "mdi:calendar-check"

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry, api_data: dict):
        """Initialize the sensor."""
        self.hass = hass
        self.entry = entry
        self.api_data = api_data

        self._attr_unique_id = util_slugify(
            f"{DOMAIN}_{entry.entry_id}_tomorrow_available"
        )
        self._attr_name = "Tomorrow Prices Available"

        self._attr_device_info = {
            "identifiers": {(DOMAIN, entry.entry_id)},
        }

    async def async_added_to_hass(self) -> None:
        """Register callbacks."""
        async_dispatcher_connect(
            self.hass, util_slugify(UPDATE_SIGNAL), self._handle_update
        )

    async def _handle_update(self) -> None:
        """Handle updated data."""
        self.async_write_ha_state()

    @property
    def is_on(self) -> bool:
        """Return true if tomorrow's prices are available."""
        return self.api_data.get("tomorrow_available", False)

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return additional attributes."""
        nordpool = self.api_data.get("nordpool")
        attrs = {}
        if nordpool:
            attrs["tomorrow_prices_count"] = len(nordpool.tomorrow)
        return attrs


class MLModelTrainedSensor(BinarySensorEntity):
    """Binary sensor indicating if ML model is trained."""

    _attr_has_entity_name = True
    _attr_device_class = BinarySensorDeviceClass.RUNNING
    _attr_icon = "mdi:brain"

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry, api_data: dict):
        """Initialize the sensor."""
        self.hass = hass
        self.entry = entry
        self.api_data = api_data

        self._attr_unique_id = util_slugify(f"{DOMAIN}_{entry.entry_id}_ml_trained")
        self._attr_name = "ML Model Trained"

        self._attr_device_info = {
            "identifiers": {(DOMAIN, entry.entry_id)},
        }

    async def async_added_to_hass(self) -> None:
        """Register callbacks."""
        async_dispatcher_connect(
            self.hass, util_slugify(UPDATE_SIGNAL), self._handle_update
        )

    async def _handle_update(self) -> None:
        """Handle updated data."""
        self.async_write_ha_state()

    @property
    def is_on(self) -> bool:
        """Return true if ML model is trained."""
        ml_predictor = self.api_data.get("ml_predictor")
        if ml_predictor:
            return ml_predictor.is_trained
        return False

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return additional attributes."""
        ml_predictor = self.api_data.get("ml_predictor")
        attrs = {}
        if ml_predictor:
            attrs["training_samples"] = ml_predictor.training_samples
            attrs["total_predictions"] = len(ml_predictor.predictions)
        return attrs
