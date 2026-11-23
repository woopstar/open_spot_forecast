"""Diagnostic sensor: the health of the data sources (#276).

Its state is how many sources' latest attempt succeeded. The attributes list
every source the update cycle used (``SourceHealth``, ``source_health.py``)
with whether it is healthy, the minutes since its last success, its last
error and how far its data reaches, plus the names of the failing ones, so
an automation or a dashboard can show which source fails and for how long.
The config entry's diagnostics (``diagnostics.py``) hold the full record.
"""

from __future__ import annotations

from typing import Any

from homeassistant.components.sensor import SensorEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import EntityCategory
from homeassistant.core import HomeAssistant
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.util import slugify as util_slugify

from .const import DOMAIN, UPDATE_SIGNAL, UPDATE_SIGNAL_FORECAST
from .source_health import SourceHealth

# The per-source values the attributes show (the diagnostics show all)
_ATTRIBUTE_KEYS = ("healthy", "age_minutes", "last_error", "data_until")


class DataSourcesSensor(SensorEntity):
    """How many data sources are healthy, with each source's status."""

    _attr_has_entity_name = True
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_translation_key = "data_sources"
    _attr_icon = "mdi:database-check"

    def __init__(
        self, hass: HomeAssistant, entry: ConfigEntry, api_data: dict[str, Any]
    ) -> None:
        """Initialize the sensor."""
        self.hass = hass
        self.api_data = api_data
        self._attr_unique_id = util_slugify(f"{DOMAIN}_{entry.entry_id}_data_sources")
        self._attr_device_info = DeviceInfo(identifiers={(DOMAIN, entry.entry_id)})

    @property
    def _health(self) -> SourceHealth | None:
        return self.api_data.get("source_health")

    async def async_added_to_hass(self) -> None:
        """Refresh after every update cycle and forecast run."""
        for signal in (UPDATE_SIGNAL, UPDATE_SIGNAL_FORECAST):
            self.async_on_remove(
                async_dispatcher_connect(
                    self.hass, util_slugify(signal), self.async_write_ha_state
                )
            )

    @property
    def native_value(self) -> int | None:
        """Return the number of healthy sources, None before any attempt."""
        health = self._health
        if health is None or not health.summary():
            return None
        return health.healthy_count

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return every source's status and the names of the failing ones."""
        summary = self._health.summary() if self._health else {}
        return {
            "sources": {
                name: {key: status[key] for key in _ATTRIBUTE_KEYS}
                for name, status in summary.items()
            },
            "failing": [
                name for name, status in summary.items() if not status["healthy"]
            ],
        }
