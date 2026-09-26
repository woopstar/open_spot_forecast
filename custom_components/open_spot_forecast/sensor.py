"""Sensor platform for Open Spot Forecast."""

import logging
from collections.abc import Sequence
from datetime import datetime, timedelta
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
from homeassistant.util import dt as dt_util, slugify as util_slugify

from .accuracy_sensor import build_lead_time_accuracy_sensors
from .attribution import ModelAttributionMixin, PriceAttributionMixin
from .const import (
    CONF_CURRENCY,
    CONF_PRECISION,
    CONF_PREDICTION_HOURS,
    CONF_PRICE_TYPE,
    CONF_REGION,
    CONF_VAT,
    DEFAULT_CURRENCY,
    DEFAULT_PRECISION,
    DEFAULT_PREDICTION_HOURS,
    DEFAULT_PRICE_TYPE,
    DEFAULT_REGION,
    DEFAULT_VAT,
    DOMAIN,
    PRICE_SOURCE_DAYAHEAD,
    SLOTS_PER_HOUR,
    UPDATE_SIGNAL,
    UPDATE_SIGNAL_FORECAST,
)
from .price_series import known_prices
from .time_slots import SLOT_MINUTES, parse_utc, slot_index_in_day

_LOGGER = logging.getLogger(__name__)


def current_prediction(
    predictions: Sequence[dict[str, Any]], now: datetime
) -> dict[str, Any] | None:
    """Return the prediction whose slot contains ``now``, else the first future one.

    Slots are compared in UTC, so the repeated hour on the DST fall-back day
    resolves to the right slot. A prediction without an ``end`` covers one
    15-minute slot; one without a parseable ``start`` is skipped.

    Args:
        predictions: Predictions with ISO ``start`` / ``end`` timestamps.
        now: The current time, timezone-aware.

    Returns:
        The prediction for the current slot; without one, the earliest
        prediction that starts after ``now``; None if every prediction is past.
    """
    first_future: dict[str, Any] | None = None
    first_future_start: datetime | None = None
    for prediction in predictions:
        start = parse_utc(prediction.get("start"))
        if start is None:
            continue
        end = parse_utc(prediction.get("end"))
        if end is None or end <= start:
            end = start + timedelta(minutes=SLOT_MINUTES)
        if start <= now < end:
            return prediction
        if now < start and (first_future_start is None or start < first_future_start):
            first_future, first_future_start = prediction, start
    return first_future


def displayed_prices(api_data: dict[str, Any], day: str) -> list[float]:
    """Return the known prices the price sensors show for ``today``/``tomorrow``.

    Stromligning's all-in consumer prices, or the day-ahead spot prices with
    VAT (#27).
    """
    if api_data.get("price_source") == PRICE_SOURCE_DAYAHEAD:
        return known_prices(api_data.get(f"prices_{day}") or [])
    stromligning_data = api_data.get("stromligning_data") or {}
    return known_prices(stromligning_data.get(day) or [])


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
    prediction_hours = entry.options.get(
        CONF_PREDICTION_HOURS,
        entry.data.get(CONF_PREDICTION_HOURS, DEFAULT_PREDICTION_HOURS),
    )

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
        MLPredictionSensor(
            hass,
            entry,
            api_data,
            currency,
            vat,
            precision,
            price_type,
            prediction_hours,
        ),
        PredictionConfidenceSensor(hass, entry, api_data),
        LearningMetricsSensor(hass, entry, api_data),
    ]
    if api_data.get("ml_predictor") is not None:
        sensors.extend(
            build_lead_time_accuracy_sensors(hass, entry, api_data, currency, precision)
        )

    async_add_entities(sensors, True)


class SpotPriceSensor(PriceAttributionMixin, SensorEntity):
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
        """Return the current price.

        Stromligning's all-in consumer price, or the day-ahead spot price of
        the current slot with VAT (#27).
        """
        if self.api_data.get("price_source") == PRICE_SOURCE_DAYAHEAD:
            prices = self.api_data.get("prices_today") or []
            index = slot_index_in_day(dt_util.now())
            price = prices[index] if index < len(prices) else None
            return None if price is None else float(round(price, self.precision))
        stromligning_data = self.api_data.get("stromligning_data")
        if stromligning_data and stromligning_data.get("current_price") is not None:
            # Stromligning already includes tariffs and VAT
            return float(round(stromligning_data["current_price"], self.precision))
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
        if self.api_data.get("price_source") == PRICE_SOURCE_DAYAHEAD:
            attrs["today_prices"] = self.api_data.get("prices_today", [])
            attrs["tomorrow_prices"] = self.api_data.get("prices_tomorrow", [])
            attrs["price_source"] = PRICE_SOURCE_DAYAHEAD
            # The day-ahead spot price with VAT; tariffs are not included
            attrs["includes_vat"] = True
            attrs["includes_tariffs"] = False
        elif stromligning_data:
            attrs["today_prices"] = stromligning_data.get("today", [])
            attrs["tomorrow_prices"] = stromligning_data.get("tomorrow", [])
            attrs["price_source"] = "stromligning"
            # The state is Stromligning's all-in consumer price
            attrs["includes_vat"] = True
            attrs["includes_tariffs"] = True

        return attrs


class TodayMinSensor(PriceAttributionMixin, SensorEntity):
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
        prices = displayed_prices(self.api_data, "today")
        return float(round(min(prices), self.precision)) if prices else None


class TodayMaxSensor(PriceAttributionMixin, SensorEntity):
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
        prices = displayed_prices(self.api_data, "today")
        return float(round(max(prices), self.precision)) if prices else None


class TodayMeanSensor(PriceAttributionMixin, SensorEntity):
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
        prices = displayed_prices(self.api_data, "today")
        return (
            float(round(sum(prices) / len(prices), self.precision)) if prices else None
        )


class TomorrowMinSensor(PriceAttributionMixin, SensorEntity):
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
        prices = displayed_prices(self.api_data, "tomorrow")
        return float(round(min(prices), self.precision)) if prices else None


class TomorrowMaxSensor(PriceAttributionMixin, SensorEntity):
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
        prices = displayed_prices(self.api_data, "tomorrow")
        return float(round(max(prices), self.precision)) if prices else None


class TomorrowMeanSensor(PriceAttributionMixin, SensorEntity):
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
        prices = displayed_prices(self.api_data, "tomorrow")
        return (
            float(round(sum(prices) / len(prices), self.precision)) if prices else None
        )


class MLPredictionSensor(ModelAttributionMixin, SensorEntity):
    """Sensor for ML-based price predictions (replaces Carnot).

    The model predicts the raw spot price excl. VAT and tariffs (#16). VAT is
    applied here, exactly once, to the state and to every price attribute;
    tariffs are not included.
    """

    _attr_has_entity_name = True
    _attr_device_class = SensorDeviceClass.MONETARY
    _attr_icon = "mdi:brain"

    def __init__(
        self,
        hass,
        entry,
        api_data,
        currency,
        vat,
        precision,
        price_type,
        prediction_hours=DEFAULT_PREDICTION_HOURS,
    ):
        self.hass = hass
        self.entry = entry
        self.api_data = api_data
        self.currency = currency
        self.vat = vat
        self.precision = precision
        self.price_type = price_type

        # Cap the predictions exposed as attributes to the configured hourly
        # window (12-hour steps, up to 72 hours) to stay under HA's 16 KB limit.
        self._max_predictions = int(prediction_hours) * SLOTS_PER_HOUR

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
        """Return the predicted price (VAT included) for the current slot.

        Falls back to the first future slot when the predictions start later.
        """
        prediction = self._state_prediction()
        if prediction is None:
            return None
        price = prediction.get("price")
        return self._with_vat(price) if price is not None else None

    def _state_prediction(self) -> dict[str, Any] | None:
        """Return the prediction the state shows (see ``current_prediction``)."""
        ml_predictor = self.api_data.get("ml_predictor")
        if not ml_predictor or not ml_predictor.predictions:
            return None
        return current_prediction(ml_predictor.predictions, dt_util.utcnow())

    def _with_vat(self, spot_price: float) -> float:
        """Return a predicted spot price (currency/kWh excl. VAT) with VAT added."""
        return float(round(spot_price * (1 + self.vat), self.precision))

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        ml_predictor = self.api_data.get("ml_predictor")
        attrs: dict[str, Any] = {}
        if ml_predictor:
            # Convert predictions to include unit of measurement. Only surface
            # the configured hourly window to stay under HA's 16 KB attribute limit.
            predictions_with_unit = []
            for pred in ml_predictor.predictions[: self._max_predictions]:
                price = pred.get("price")
                if price is not None:
                    predictions_with_unit.append(
                        {
                            "start": pred.get("start"),
                            "end": pred.get("end"),
                            "price": self._with_vat(price),
                            "unit": f"{self.currency}/{self.price_type}",
                            "confidence": pred.get("confidence"),
                        }
                    )
            attrs["predictions"] = predictions_with_unit
            # The slot whose prediction is the state
            state_prediction = self._state_prediction()
            attrs["state_slot_start"] = (
                state_prediction.get("start") if state_prediction else None
            )
            # Every price above and below: spot price + VAT, no tariffs
            attrs["includes_vat"] = True
            attrs["includes_tariffs"] = False
            attrs["vat"] = self.vat
            stats = ml_predictor.get_prediction_stats()
            if stats:
                # Predictions are already in currency/kWh: only VAT is added
                attrs["forecast_min"] = self._with_vat(stats.get("min_price", 0))
                attrs["forecast_max"] = self._with_vat(stats.get("max_price", 0))
                attrs["forecast_mean"] = self._with_vat(stats.get("mean_price", 0))
                attrs["unit"] = f"{self.currency}/{self.price_type}"
                attrs["mean_confidence"] = stats.get("mean_confidence")
                attrs["total_predictions"] = stats.get("total_predictions")
                attrs["is_ml_model"] = stats.get("is_ml_model")
                attrs["training_samples"] = stats.get("training_samples")
        return attrs


class PredictionConfidenceSensor(ModelAttributionMixin, SensorEntity):
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


class LearningMetricsSensor(ModelAttributionMixin, SensorEntity):
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
        metrics = self._cached_metrics
        if metrics is None:
            ml_predictor = self.api_data.get("ml_predictor")
            metrics = ml_predictor.get_learning_metrics() if ml_predictor else {}
            self._cached_metrics = metrics
        return metrics

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
            # Holdout error of the latest training (None before one)
            attrs["holdout_mae"] = metrics.get("holdout_mae")
            attrs["holdout_rmse"] = metrics.get("holdout_rmse")
            attrs["holdout_trained_at"] = metrics.get("holdout_trained_at")
        return attrs
