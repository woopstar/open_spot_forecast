"""Diagnostic sensors for live ML forecast accuracy per lead time."""

from typing import Any

from homeassistant.components.sensor import SensorEntity, SensorStateClass
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import EntityCategory, UnitOfEnergy
from homeassistant.core import HomeAssistant
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.util import slugify as util_slugify

from .const import DOMAIN, LEAD_TIME_BUCKETS, LEAD_TIME_WINDOW_DAYS, UPDATE_SIGNAL

LEAD_TIME_METRICS = ("mae", "rmse")


def build_lead_time_accuracy_sensors(
    hass: HomeAssistant,
    entry: ConfigEntry,
    api_data: dict[str, Any],
    currency: str,
    precision: int,
) -> list[LeadTimeAccuracySensor]:
    """Return one MAE and one RMSE sensor per lead-time bucket.

    Args:
        hass: Home Assistant instance.
        entry: Config entry the sensors belong to.
        api_data: Integration data holding the ``ml_predictor``.
        currency: Currency of the price data.
        precision: Suggested display precision.

    Returns:
        The accuracy sensors, ordered by bucket then metric.
    """
    return [
        LeadTimeAccuracySensor(
            hass, entry, api_data, currency, precision, bucket, metric
        )
        for bucket, _ in LEAD_TIME_BUCKETS
        for metric in LEAD_TIME_METRICS
    ]


class LeadTimeAccuracySensor(SensorEntity):
    """Live MAE or RMSE of the ML forecast for one lead-time bucket."""

    _attr_has_entity_name = True
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_icon = "mdi:target"

    def __init__(
        self,
        hass: HomeAssistant,
        entry: ConfigEntry,
        api_data: dict[str, Any],
        currency: str,
        precision: int,
        bucket: str,
        metric: str,
    ) -> None:
        """Initialize the sensor for a ``LEAD_TIME_BUCKETS`` key and metric."""
        self.hass = hass
        self.api_data = api_data
        self.bucket = bucket
        self.metric = metric

        key = f"forecast_{metric}_{bucket}"
        self._attr_translation_key = key
        self._attr_unique_id = util_slugify(f"{DOMAIN}_{entry.entry_id}_{key}")
        # Errors are in the unit of the confirmed prices the model learns from.
        self._attr_native_unit_of_measurement = (
            f"{currency}/{UnitOfEnergy.KILO_WATT_HOUR}"
        )
        self._attr_suggested_display_precision = precision
        self._attr_device_info = {"identifiers": {(DOMAIN, entry.entry_id)}}

    async def async_added_to_hass(self) -> None:
        """Refresh the state after every self-learning update."""
        self.async_on_remove(
            async_dispatcher_connect(
                self.hass, util_slugify(UPDATE_SIGNAL), self.async_write_ha_state
            )
        )

    def _bucket_stats(self) -> dict[str, float | int]:
        """Return the cached accuracy summary for this sensor's bucket."""
        ml_predictor = self.api_data.get("ml_predictor")
        if ml_predictor is None:
            return {}
        stats: dict[str, float | int] = ml_predictor.lead_time_accuracy.get(
            self.bucket, {}
        )
        return stats

    @property
    def native_value(self) -> float | None:
        """Return the MAE or RMSE, or None until a prediction has been matched."""
        value = self._bucket_stats().get(self.metric)
        return float(value) if value is not None else None

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return the sample count, mean bias and rolling window length."""
        stats = self._bucket_stats()
        return {
            "samples": stats.get("samples", 0),
            "bias": stats.get("bias"),
            "window_days": LEAD_TIME_WINDOW_DAYS,
        }
