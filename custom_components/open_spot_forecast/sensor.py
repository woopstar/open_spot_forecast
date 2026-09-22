"""Sensor platform for Open Spot Forecast."""

import logging
from typing import Any

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.util import slugify as util_slugify

from .const import (
    CONF_CURRENCY,
    CONF_PRECISION,
    CONF_PRICE_TYPE,
    CONF_REGION,
    CONF_VAT,
    DEFAULT_CURRENCY,
    DEFAULT_PRECISION,
    DEFAULT_PRICE_TYPE,
    DEFAULT_REGION,
    DEFAULT_VAT,
    DOMAIN,
    PRICE_IN,
    UPDATE_SIGNAL,
    UPDATE_SIGNAL_FORECAST,
)

_LOGGER = logging.getLogger(__name__)

# Cap the number of predictions exposed as entity attributes. The full 7-day
# forecast (672 slots) blows past Home Assistant's 16 KB attribute limit and
# slows down state writes, so we only surface the next 24 hours (96 slots).
_MAX_PREDICTIONS_IN_ATTRIBUTES = 96


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Open Spot Forecast sensors."""
    api_data = hass.data[DOMAIN][entry.entry_id]

    region = entry.data.get(CONF_REGION, DEFAULT_REGION)
    currency = entry.data.get(CONF_CURRENCY, DEFAULT_CURRENCY)
    vat = entry.options.get(CONF_VAT, DEFAULT_VAT)
    precision = entry.options.get(CONF_PRECISION, DEFAULT_PRECISION)
    price_type = entry.options.get(CONF_PRICE_TYPE, DEFAULT_PRICE_TYPE)

    sensors = [
        SpotPriceSensor(
            hass, entry, api_data, region, currency, vat, precision, price_type
        ),
        TodayMinSensor(hass, entry, api_data, currency, vat, precision, price_type),
        TodayMaxSensor(hass, entry, api_data, currency, vat, precision, price_type),
        TodayMeanSensor(hass, entry, api_data, currency, vat, precision, price_type),
        TomorrowMinSensor(hass, entry, api_data, currency, vat, precision, price_type),
        TomorrowMaxSensor(hass, entry, api_data, currency, vat, precision, price_type),
        TomorrowMeanSensor(hass, entry, api_data, currency, vat, precision, price_type),
        MLPredictionSensor(hass, entry, api_data, currency, vat, precision, price_type),
        PredictionConfidenceSensor(hass, entry, api_data),
        LearningMetricsSensor(hass, entry, api_data),
    ]

    async_add_entities(sensors, True)


class SpotPriceSensor(SensorEntity):
    """Sensor for current spot price."""

    _attr_has_entity_name = True
    _attr_device_class = SensorDeviceClass.MONETARY
    _attr_state_class = SensorStateClass.TOTAL
    _attr_icon = "mdi:flash"

    def __init__(
        self,
        hass: HomeAssistant,
        entry: ConfigEntry,
        api_data: dict,
        region: str,
        currency: str,
        vat: float,
        precision: int,
        price_type: str,
    ):
        """Initialize the sensor."""
        self.hass = hass
        self.entry = entry
        self.api_data = api_data
        self.region = region
        self.currency = currency
        self.vat = vat
        self.precision = precision
        self.price_type = price_type

        self._attr_unique_id = util_slugify(f"{DOMAIN}_{entry.entry_id}_current_price")
        self._attr_name = "Current Spot Price"
        self._attr_native_unit_of_measurement = f"{currency}/{price_type}"
        self._attr_suggested_display_precision = precision

        self._attr_device_info = {
            "identifiers": {(DOMAIN, entry.entry_id)},
            "name": f"Open Spot Forecast {region}",
            "manufacturer": "Open Spot Forecast",
            "model": "Spot Price Predictor",
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
    def native_value(self) -> float | None:
        """Return the current price."""
        # Priority: Stromligning (real price with tariffs/VAT) > Nordpool > API
        stromligning_data = self.api_data.get("stromligning_data")
        if stromligning_data and stromligning_data.get("current_price") is not None:
            # Stromligning already includes tariffs and VAT
            return float(round(stromligning_data["current_price"], self.precision))

        nordpool = self.api_data.get("nordpool")
        if nordpool:
            price = nordpool.get_current_price()
            if price is not None:
                # Convert from MWh to kWh and apply VAT
                converted = price / PRICE_IN.get(self.price_type, 1000)
                return float(round(converted * (1 + self.vat), self.precision))
        return None

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return additional attributes."""
        attrs = {
            "region": self.region,
            "currency": self.currency,
            "vat": self.vat,
            "last_update": self.api_data.get("last_update"),
        }

        # Include Stromligning 15-min prices if available. We deliberately omit
        # the raw dict arrays (prices_15min / raw_today / raw_tomorrow) — they
        # are large and push the attribute payload past HA's 16 KB limit.
        stromligning_data = self.api_data.get("stromligning_data")
        if stromligning_data:
            attrs["today_prices"] = stromligning_data.get("today", [])
            attrs["tomorrow_prices"] = stromligning_data.get("tomorrow", [])
            attrs["price_source"] = "stromligning"
        else:
            # Fall back to Nordpool
            nordpool = self.api_data.get("nordpool")
            if nordpool:
                attrs["today_prices"] = nordpool.today
                attrs["tomorrow_prices"] = nordpool.tomorrow
                attrs["price_source"] = "nordpool"

        return attrs


class TodayMinSensor(SensorEntity):
    """Sensor for today's minimum price."""

    _attr_has_entity_name = True
    _attr_device_class = SensorDeviceClass.MONETARY
    _attr_icon = "mdi:trending-down"

    def __init__(self, hass, entry, api_data, currency, vat, precision, price_type):
        self.hass = hass
        self.entry = entry
        self.api_data = api_data
        self.currency = currency
        self.vat = vat
        self.precision = precision
        self.price_type = price_type

        self._attr_unique_id = util_slugify(f"{DOMAIN}_{entry.entry_id}_today_min")
        self._attr_name = "Today Min Price"
        self._attr_native_unit_of_measurement = f"{currency}/{price_type}"
        self._attr_suggested_display_precision = precision

        self._attr_device_info = {
            "identifiers": {(DOMAIN, entry.entry_id)},
        }

    async def async_added_to_hass(self) -> None:
        async_dispatcher_connect(
            self.hass, util_slugify(UPDATE_SIGNAL), self._handle_update
        )

    async def _handle_update(self) -> None:
        self.async_write_ha_state()

    @property
    def native_value(self) -> float | None:
        # Try Stromligning first (priority)
        stromligning_data = self.api_data.get("stromligning_data")
        if stromligning_data and stromligning_data.get("today"):
            prices = stromligning_data["today"]
            if prices:
                # Stromligning prices already include VAT and tariffs
                min_price = min(prices)
                return float(round(min_price, self.precision))

        # Fallback to Nordpool
        nordpool = self.api_data.get("nordpool")
        if nordpool:
            stats = nordpool.get_today_stats()
            if stats and "min" in stats:
                converted = stats["min"] / PRICE_IN.get(self.price_type, 1000)
                return float(round(converted * (1 + self.vat), self.precision))
        return None


class TodayMaxSensor(SensorEntity):
    """Sensor for today's maximum price."""

    _attr_has_entity_name = True
    _attr_device_class = SensorDeviceClass.MONETARY
    _attr_icon = "mdi:trending-up"

    def __init__(self, hass, entry, api_data, currency, vat, precision, price_type):
        self.hass = hass
        self.entry = entry
        self.api_data = api_data
        self.currency = currency
        self.vat = vat
        self.precision = precision
        self.price_type = price_type

        self._attr_unique_id = util_slugify(f"{DOMAIN}_{entry.entry_id}_today_max")
        self._attr_name = "Today Max Price"
        self._attr_native_unit_of_measurement = f"{currency}/{price_type}"
        self._attr_suggested_display_precision = precision

        self._attr_device_info = {
            "identifiers": {(DOMAIN, entry.entry_id)},
        }

    async def async_added_to_hass(self) -> None:
        async_dispatcher_connect(
            self.hass, util_slugify(UPDATE_SIGNAL), self._handle_update
        )

    async def _handle_update(self) -> None:
        self.async_write_ha_state()

    @property
    def native_value(self) -> float | None:
        # Try Stromligning first (priority)
        stromligning_data = self.api_data.get("stromligning_data")
        if stromligning_data and stromligning_data.get("today"):
            prices = stromligning_data["today"]
            if prices:
                # Stromligning prices already include VAT and tariffs
                max_price = max(prices)
                return float(round(max_price, self.precision))

        # Fallback to Nordpool
        nordpool = self.api_data.get("nordpool")
        if nordpool:
            stats = nordpool.get_today_stats()
            if stats and "max" in stats:
                converted = stats["max"] / PRICE_IN.get(self.price_type, 1000)
                return float(round(converted * (1 + self.vat), self.precision))
        return None


class TodayMeanSensor(SensorEntity):
    """Sensor for today's mean price."""

    _attr_has_entity_name = True
    _attr_device_class = SensorDeviceClass.MONETARY
    _attr_icon = "mdi:chart-line"

    def __init__(self, hass, entry, api_data, currency, vat, precision, price_type):
        self.hass = hass
        self.entry = entry
        self.api_data = api_data
        self.currency = currency
        self.vat = vat
        self.precision = precision
        self.price_type = price_type

        self._attr_unique_id = util_slugify(f"{DOMAIN}_{entry.entry_id}_today_mean")
        self._attr_name = "Today Mean Price"
        self._attr_native_unit_of_measurement = f"{currency}/{price_type}"
        self._attr_suggested_display_precision = precision

        self._attr_device_info = {
            "identifiers": {(DOMAIN, entry.entry_id)},
        }

    async def async_added_to_hass(self) -> None:
        async_dispatcher_connect(
            self.hass, util_slugify(UPDATE_SIGNAL), self._handle_update
        )

    async def _handle_update(self) -> None:
        self.async_write_ha_state()

    @property
    def native_value(self) -> float | None:
        # Try Stromligning first (priority)
        stromligning_data = self.api_data.get("stromligning_data")
        if stromligning_data and stromligning_data.get("today"):
            prices = stromligning_data["today"]
            if prices:
                # Stromligning prices already include VAT and tariffs
                mean_price = sum(prices) / len(prices)
                return float(round(mean_price, self.precision))

        # Fallback to Nordpool
        nordpool = self.api_data.get("nordpool")
        if nordpool:
            stats = nordpool.get_today_stats()
            if stats and "mean" in stats:
                converted = stats["mean"] / PRICE_IN.get(self.price_type, 1000)
                return float(round(converted * (1 + self.vat), self.precision))
        return None


class TomorrowMinSensor(SensorEntity):
    """Sensor for tomorrow's minimum price."""

    _attr_has_entity_name = True
    _attr_device_class = SensorDeviceClass.MONETARY
    _attr_icon = "mdi:trending-down"

    def __init__(self, hass, entry, api_data, currency, vat, precision, price_type):
        self.hass = hass
        self.entry = entry
        self.api_data = api_data
        self.currency = currency
        self.vat = vat
        self.precision = precision
        self.price_type = price_type

        self._attr_unique_id = util_slugify(f"{DOMAIN}_{entry.entry_id}_tomorrow_min")
        self._attr_name = "Tomorrow Min Price"
        self._attr_native_unit_of_measurement = f"{currency}/{price_type}"
        self._attr_suggested_display_precision = precision

        self._attr_device_info = {
            "identifiers": {(DOMAIN, entry.entry_id)},
        }

    async def async_added_to_hass(self) -> None:
        async_dispatcher_connect(
            self.hass, util_slugify(UPDATE_SIGNAL), self._handle_update
        )

    async def _handle_update(self) -> None:
        self.async_write_ha_state()

    @property
    def native_value(self) -> float | None:
        # Try Stromligning first (priority)
        stromligning_data = self.api_data.get("stromligning_data")
        if stromligning_data and stromligning_data.get("tomorrow"):
            prices = stromligning_data["tomorrow"]
            if prices:
                # Stromligning prices already include VAT and tariffs
                min_price = min(prices)
                return float(round(min_price, self.precision))

        # Fallback to Nordpool
        nordpool = self.api_data.get("nordpool")
        if nordpool:
            stats = nordpool.get_tomorrow_stats()
            if stats and "min" in stats:
                converted = stats["min"] / PRICE_IN.get(self.price_type, 1000)
                return float(round(converted * (1 + self.vat), self.precision))
        return None


class TomorrowMaxSensor(SensorEntity):
    """Sensor for tomorrow's maximum price."""

    _attr_has_entity_name = True
    _attr_device_class = SensorDeviceClass.MONETARY
    _attr_icon = "mdi:trending-up"

    def __init__(self, hass, entry, api_data, currency, vat, precision, price_type):
        self.hass = hass
        self.entry = entry
        self.api_data = api_data
        self.currency = currency
        self.vat = vat
        self.precision = precision
        self.price_type = price_type

        self._attr_unique_id = util_slugify(f"{DOMAIN}_{entry.entry_id}_tomorrow_max")
        self._attr_name = "Tomorrow Max Price"
        self._attr_native_unit_of_measurement = f"{currency}/{price_type}"
        self._attr_suggested_display_precision = precision

        self._attr_device_info = {
            "identifiers": {(DOMAIN, entry.entry_id)},
        }

    async def async_added_to_hass(self) -> None:
        async_dispatcher_connect(
            self.hass, util_slugify(UPDATE_SIGNAL), self._handle_update
        )

    async def _handle_update(self) -> None:
        self.async_write_ha_state()

    @property
    def native_value(self) -> float | None:
        # Try Stromligning first (priority)
        stromligning_data = self.api_data.get("stromligning_data")
        if stromligning_data and stromligning_data.get("tomorrow"):
            prices = stromligning_data["tomorrow"]
            if prices:
                # Stromligning prices already include VAT and tariffs
                max_price = max(prices)
                return float(round(max_price, self.precision))

        # Fallback to Nordpool
        nordpool = self.api_data.get("nordpool")
        if nordpool:
            stats = nordpool.get_tomorrow_stats()
            if stats and "max" in stats:
                converted = stats["max"] / PRICE_IN.get(self.price_type, 1000)
                return float(round(converted * (1 + self.vat), self.precision))
        return None


class TomorrowMeanSensor(SensorEntity):
    """Sensor for tomorrow's mean price."""

    _attr_has_entity_name = True
    _attr_device_class = SensorDeviceClass.MONETARY
    _attr_icon = "mdi:chart-line"

    def __init__(self, hass, entry, api_data, currency, vat, precision, price_type):
        self.hass = hass
        self.entry = entry
        self.api_data = api_data
        self.currency = currency
        self.vat = vat
        self.precision = precision
        self.price_type = price_type

        self._attr_unique_id = util_slugify(f"{DOMAIN}_{entry.entry_id}_tomorrow_mean")
        self._attr_name = "Tomorrow Mean Price"
        self._attr_native_unit_of_measurement = f"{currency}/{price_type}"
        self._attr_suggested_display_precision = precision

        self._attr_device_info = {
            "identifiers": {(DOMAIN, entry.entry_id)},
        }

    async def async_added_to_hass(self) -> None:
        async_dispatcher_connect(
            self.hass, util_slugify(UPDATE_SIGNAL), self._handle_update
        )

    async def _handle_update(self) -> None:
        self.async_write_ha_state()

    @property
    def native_value(self) -> float | None:
        # Try Stromligning first (priority)
        stromligning_data = self.api_data.get("stromligning_data")
        if stromligning_data and stromligning_data.get("tomorrow"):
            prices = stromligning_data["tomorrow"]
            if prices:
                # Stromligning prices already include VAT and tariffs
                mean_price = sum(prices) / len(prices)
                return float(round(mean_price, self.precision))

        # Fallback to Nordpool
        nordpool = self.api_data.get("nordpool")
        if nordpool:
            stats = nordpool.get_tomorrow_stats()
            if stats and "mean" in stats:
                converted = stats["mean"] / PRICE_IN.get(self.price_type, 1000)
                return float(round(converted * (1 + self.vat), self.precision))
        return None


class MLPredictionSensor(SensorEntity):
    """Sensor for ML-based price predictions (replaces Carnot)."""

    _attr_has_entity_name = True
    _attr_device_class = SensorDeviceClass.MONETARY
    _attr_icon = "mdi:brain"

    def __init__(self, hass, entry, api_data, currency, vat, precision, price_type):
        self.hass = hass
        self.entry = entry
        self.api_data = api_data
        self.currency = currency
        self.vat = vat
        self.precision = precision
        self.price_type = price_type

        self._attr_unique_id = util_slugify(f"{DOMAIN}_{entry.entry_id}_ml_prediction")
        self._attr_name = "Price Forecast (ML)"
        self._attr_native_unit_of_measurement = f"{currency}/{price_type}"
        self._attr_suggested_display_precision = precision

        self._attr_device_info = {
            "identifiers": {(DOMAIN, entry.entry_id)},
        }

    async def async_added_to_hass(self) -> None:
        async_dispatcher_connect(
            self.hass, util_slugify(UPDATE_SIGNAL_FORECAST), self._handle_update
        )

    async def _handle_update(self) -> None:
        self.async_write_ha_state()

    @property
    def native_value(self) -> float | None:
        ml_predictor = self.api_data.get("ml_predictor")
        if ml_predictor:
            predictions = ml_predictor.predictions
            if predictions:
                # Find the next prediction (closest future timestamp)
                from datetime import datetime

                now = datetime.now()
                next_pred = None
                for pred in predictions:
                    start_str = pred.get("start")
                    if start_str:
                        try:
                            pred_time = datetime.fromisoformat(start_str)
                            if pred_time > now:
                                next_pred = pred
                                break
                        except ValueError, TypeError:
                            continue

                # Fallback to first prediction if no future prediction found
                if next_pred is None:
                    next_pred = predictions[0]

                price = next_pred.get("price")
                if price is not None:
                    # Prices are already in kr/kWh, just apply VAT
                    return float(round(price * (1 + self.vat), self.precision))
        return None

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        ml_predictor = self.api_data.get("ml_predictor")
        attrs: dict[str, Any] = {}
        if ml_predictor:
            # Convert predictions to include unit of measurement. Only surface
            # the next 24 hours to stay under HA's 16 KB attribute limit.
            predictions_with_unit = []
            for pred in ml_predictor.predictions[:_MAX_PREDICTIONS_IN_ATTRIBUTES]:
                price = pred.get("price")
                if price is not None:
                    # Prices are already in kr/kWh, just apply VAT
                    predictions_with_unit.append(
                        {
                            "start": pred.get("start"),
                            "end": pred.get("end"),
                            "price": round(price * (1 + self.vat), self.precision),
                            "unit": f"{self.currency}/{self.price_type}",
                            "confidence": pred.get("confidence"),
                        }
                    )
            attrs["predictions"] = predictions_with_unit
            stats = ml_predictor.get_prediction_stats()
            if stats:
                # Convert stats to proper unit
                conversion_factor = (
                    1 / PRICE_IN.get(self.price_type, 1000) * (1 + self.vat)
                )
                attrs["forecast_min"] = round(
                    stats.get("min_price", 0) * conversion_factor, self.precision
                )
                attrs["forecast_max"] = round(
                    stats.get("max_price", 0) * conversion_factor, self.precision
                )
                attrs["forecast_mean"] = round(
                    stats.get("mean_price", 0) * conversion_factor, self.precision
                )
                attrs["unit"] = f"{self.currency}/{self.price_type}"
                attrs["mean_confidence"] = stats.get("mean_confidence")
                attrs["total_predictions"] = stats.get("total_predictions")
                attrs["is_ml_model"] = stats.get("is_ml_model")
                attrs["training_samples"] = stats.get("training_samples")
        return attrs


class PredictionConfidenceSensor(SensorEntity):
    """Sensor for prediction confidence score."""

    _attr_has_entity_name = True
    _attr_icon = "mdi:gauge"

    def __init__(self, hass, entry, api_data):
        self.hass = hass
        self.entry = entry
        self.api_data = api_data

        self._attr_unique_id = util_slugify(f"{DOMAIN}_{entry.entry_id}_confidence")
        self._attr_name = "Prediction Confidence"
        self._attr_native_unit_of_measurement = "%"
        self._attr_suggested_display_precision = 1

        self._attr_device_info = {
            "identifiers": {(DOMAIN, entry.entry_id)},
        }

    async def async_added_to_hass(self) -> None:
        async_dispatcher_connect(
            self.hass, util_slugify(UPDATE_SIGNAL_FORECAST), self._handle_update
        )

    async def _handle_update(self) -> None:
        self.async_write_ha_state()

    @property
    def native_value(self) -> float | None:
        ml_predictor = self.api_data.get("ml_predictor")
        if ml_predictor:
            stats = ml_predictor.get_prediction_stats()
            if stats and "mean_confidence" in stats:
                return float(round(stats["mean_confidence"] * 100, 1))
        return None


class LearningMetricsSensor(SensorEntity):
    """Sensor for self-learning metrics and error tracking."""

    _attr_has_entity_name = True
    _attr_icon = "mdi:school"

    def __init__(self, hass, entry, api_data):
        self.hass = hass
        self.entry = entry
        self.api_data = api_data
        self._cached_metrics: dict[str, Any] | None = None

        self._attr_unique_id = util_slugify(
            f"{DOMAIN}_{entry.entry_id}_learning_metrics"
        )
        self._attr_name = "Learning Metrics"
        self._attr_native_unit_of_measurement = "samples"

        self._attr_device_info = {
            "identifiers": {(DOMAIN, entry.entry_id)},
        }

    async def async_added_to_hass(self) -> None:
        async_dispatcher_connect(
            self.hass, util_slugify(UPDATE_SIGNAL), self._handle_update
        )

    async def _handle_update(self) -> None:
        # Invalidate the cache so the next state write recomputes metrics.
        self._cached_metrics = None
        self.async_write_ha_state()

    def _get_metrics(self) -> dict[str, Any]:
        """Return learning metrics, computing once per state write.

        ``native_value`` and ``extra_state_attributes`` are both evaluated
        during a single state write; caching avoids running the (expensive)
        metric aggregation twice and keeps the update under HA's 0.5 s
        slow-update threshold.
        """
        if self._cached_metrics is None:
            ml_predictor = self.api_data.get("ml_predictor")
            self._cached_metrics = (
                ml_predictor.get_learning_metrics() if ml_predictor else {}
            )
        return self._cached_metrics

    @property
    def native_value(self) -> int | None:
        metrics = self._get_metrics()
        return metrics.get("total_samples", 0) if metrics else None

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        metrics = self._get_metrics()
        attrs = {}
        if metrics:
            attrs["status"] = metrics.get("status", "idle")
            attrs["message"] = metrics.get("message", "")
            attrs["is_learning"] = metrics.get("is_learning", False)
            attrs["mae"] = metrics.get("mae")
            attrs["rmse"] = metrics.get("rmse")
            attrs["mean_bias"] = metrics.get("mean_bias")
            attrs["mean_pct_error"] = metrics.get("mean_pct_error")
            attrs["learning_confidence"] = metrics.get("learning_confidence")
            attrs["hours_tracked"] = metrics.get("slots_tracked")
            attrs["bias_corrections"] = metrics.get("bias_corrections")
            attrs["pending_predictions"] = metrics.get("pending_predictions")
            attrs["hourly_metrics"] = metrics.get("hourly_metrics")
        return attrs
