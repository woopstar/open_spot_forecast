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
    ATTRIBUTE_FORMAT_COMPACT,
    CONF_ATTRIBUTE_FORMAT,
    CONF_INCLUDE_KNOWN_PRICES,
    CONF_PREDBAT_SENSORS,
    CONF_PREDICTION_HOURS,
    CONF_REGION,
    DEFAULT_ATTRIBUTE_FORMAT,
    DEFAULT_INCLUDE_KNOWN_PRICES,
    DEFAULT_PREDBAT_SENSORS,
    DEFAULT_PREDICTION_HOURS,
    DEFAULT_REGION,
    DETAILED_MAX_PREDICTION_HOURS,
    DOMAIN,
    PRICE_SOURCE_DAYAHEAD,
    UPDATE_SIGNAL,
    UPDATE_SIGNAL_FORECAST,
)
from .day_ahead_sensor import DayAheadPredictionSensor
from .evaluation_sensor import ForecastEvaluationSensor
from .forecast_attributes import compact_forecast, detailed_forecast, fit_compact
from .predbat_sensor import build_predbat_sensors
from .price_output import HOUR_MINUTES, PriceOutput
from .price_series import known_prices
from .price_source import PriceSettings
from .spot_prices import known_until, with_known_prices
from .tariffs import TariffSchedule
from .time_slots import SLOT_MINUTES, parse_utc

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


def source_day_prices(api_data: dict[str, Any], day: str) -> list[float | None]:
    """Return a day's prices excl. VAT as the price source delivers them.

    Args:
        api_data: Integration data.
        day: ``today`` or ``tomorrow``.

    Returns:
        One price per 15-min slot from local midnight (None if missing):
        Stromligning's consumer price (tariffs included, #107) or the
        day-ahead spot price (#27), both excl. VAT.
    """
    if api_data.get("price_source") == PRICE_SOURCE_DAYAHEAD:
        return list(api_data.get(f"prices_{day}") or [])
    stromligning_data = api_data.get("stromligning_data") or {}
    return list(stromligning_data.get(day) or [])


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Open Spot Forecast sensors."""
    api_data = hass.data[DOMAIN][entry.entry_id]

    region = entry.data.get(CONF_REGION, DEFAULT_REGION)
    settings = PriceSettings.from_entry(entry)
    currency, output = settings.currency, settings.output
    prediction_hours = entry.options.get(
        CONF_PREDICTION_HOURS,
        entry.data.get(CONF_PREDICTION_HOURS, DEFAULT_PREDICTION_HOURS),
    )
    attribute_format = entry.options.get(
        CONF_ATTRIBUTE_FORMAT, DEFAULT_ATTRIBUTE_FORMAT
    )
    include_known = entry.options.get(
        CONF_INCLUDE_KNOWN_PRICES, DEFAULT_INCLUDE_KNOWN_PRICES
    )

    sensors = [
        SpotPriceSensor(hass, entry, api_data, region, currency, output),
        TodayMinSensor(hass, entry, api_data, currency, output),
        TodayMaxSensor(hass, entry, api_data, currency, output),
        TodayMeanSensor(hass, entry, api_data, currency, output),
        TomorrowMinSensor(hass, entry, api_data, currency, output),
        TomorrowMaxSensor(hass, entry, api_data, currency, output),
        TomorrowMeanSensor(hass, entry, api_data, currency, output),
        MLPredictionSensor(
            hass,
            entry,
            api_data,
            currency,
            output,
            prediction_hours,
            attribute_format,
            include_known,
        ),
        PredictionConfidenceSensor(hass, entry, api_data),
        LearningMetricsSensor(hass, entry, api_data),
    ]
    if api_data.get("ml_predictor") is not None:
        sensors.extend(
            build_lead_time_accuracy_sensors(
                hass, entry, api_data, currency, output.precision
            )
        )
        sensors.append(
            ForecastEvaluationSensor(hass, entry, api_data, currency, output)
        )
        # The day-ahead prediction of the current slot, for the recorder (#113)
        sensors.append(
            DayAheadPredictionSensor(hass, entry, api_data, currency, output)
        )
    if entry.options.get(CONF_PREDBAT_SENSORS, DEFAULT_PREDBAT_SENSORS):
        # Predbat's import/export, today/tomorrow rate entities (#124)
        sensors.extend(build_predbat_sensors(hass, entry, api_data, currency, output))

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
        output: PriceOutput,
    ):
        """Initialize the sensor."""
        self.hass = hass
        self.entry = entry
        self.api_data = api_data
        self.region = region
        self.currency = currency
        self.output = output

        self._attr_unique_id = util_slugify(f"{DOMAIN}_{entry.entry_id}_current_price")
        self._attr_name = "Current Spot Price"
        self._attr_native_unit_of_measurement = output.unit(currency)
        self._attr_suggested_display_precision = output.precision

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

        Stromligning's consumer price or the day-ahead spot price of the
        current slot, with the surcharge and VAT (#27, #39, #107). With
        ``hourly_average`` it is the current local hour's mean.
        """
        stromligning_data = self.api_data.get("stromligning_data") or {}
        dayahead = self.api_data.get("price_source") == PRICE_SOURCE_DAYAHEAD
        if not dayahead and not self.output.hourly_average:
            current = stromligning_data.get("current_price")
            return None if current is None else self.output.convert(current)
        prices = source_day_prices(self.api_data, "today")
        return self.output.price_at(prices, dt_util.now())

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return additional attributes."""
        attrs = {
            "region": self.region,
            "currency": self.currency,
            "vat": self.output.vat,
            "hourly_average": self.output.hourly_average,
            "last_update": self.api_data.get("last_update"),
        }

        # Today's and tomorrow's prices per slot (per hour with
        # hourly_average). The raw dict arrays (prices_15min / raw_today /
        # raw_tomorrow) are omitted: they push the payload past HA's 16 KB limit
        if self.api_data.get("price_source") == PRICE_SOURCE_DAYAHEAD:
            attrs["price_source"] = PRICE_SOURCE_DAYAHEAD
            # The day-ahead spot price: no tariffs
            attrs["includes_tariffs"] = False
        elif self.api_data.get("stromligning_data"):
            attrs["price_source"] = "stromligning"
            # Stromligning's consumer price: tariffs, tax and fees included
            attrs["includes_tariffs"] = True
        else:
            return attrs
        # Every source excl. VAT, then (price + surcharge) × (1 + VAT) (#107)
        attrs["surcharge"] = self.output.surcharge
        attrs["includes_vat"] = True
        for day in ("today", "tomorrow"):
            attrs[f"{day}_prices"] = self.output.day_prices(
                source_day_prices(self.api_data, day)
            )
        return attrs


class DayPriceStatSensor(PriceAttributionMixin, SensorEntity):
    """Today's or tomorrow's minimum, maximum or mean price.

    The statistic is taken over the day's known prices per slot (per hour
    with ``hourly_average``), then converted like every exposed price.
    """

    _attr_has_entity_name = True
    _attr_device_class = SensorDeviceClass.MONETARY
    # "today" or "tomorrow", and "min", "max" or "mean"
    _day: str
    _stat: str

    def __init__(
        self,
        hass: HomeAssistant,
        entry: ConfigEntry,
        api_data: dict,
        currency: str,
        output: PriceOutput,
    ):
        self.hass = hass
        self.entry = entry
        self.api_data = api_data
        self.currency = currency
        self.output = output

        self._attr_unique_id = util_slugify(
            f"{DOMAIN}_{entry.entry_id}_{self._day}_{self._stat}"
        )
        self._attr_name = f"{self._day.capitalize()} {self._stat.capitalize()} Price"
        self._attr_native_unit_of_measurement = output.unit(currency)
        self._attr_suggested_display_precision = output.precision

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
        prices = source_day_prices(self.api_data, self._day)
        known = known_prices(self.output.day_series(prices))
        if not known:
            return None
        if self._stat == "min":
            value = min(known)
        elif self._stat == "max":
            value = max(known)
        else:
            value = sum(known) / len(known)
        return self.output.convert(value)


class TodayMinSensor(DayPriceStatSensor):
    """Sensor for today's minimum price."""

    _day, _stat = "today", "min"
    _attr_icon = "mdi:trending-down"


class TodayMaxSensor(DayPriceStatSensor):
    """Sensor for today's maximum price."""

    _day, _stat = "today", "max"
    _attr_icon = "mdi:trending-up"


class TodayMeanSensor(DayPriceStatSensor):
    """Sensor for today's mean price."""

    _day, _stat = "today", "mean"
    _attr_icon = "mdi:chart-line"


class TomorrowMinSensor(DayPriceStatSensor):
    """Sensor for tomorrow's minimum price."""

    _day, _stat = "tomorrow", "min"
    _attr_icon = "mdi:trending-down"


class TomorrowMaxSensor(DayPriceStatSensor):
    """Sensor for tomorrow's maximum price."""

    _day, _stat = "tomorrow", "max"
    _attr_icon = "mdi:trending-up"


class TomorrowMeanSensor(DayPriceStatSensor):
    """Sensor for tomorrow's mean price."""

    _day, _stat = "tomorrow", "mean"
    _attr_icon = "mdi:chart-line"


class MLPredictionSensor(ModelAttributionMixin, SensorEntity):
    """Sensor for ML-based price predictions (replaces Carnot).

    The model predicts the raw spot price excl. VAT and tariffs (#16). Each
    slot's tariff (``api_data["tariffs"]``, #107), the surcharge and VAT are
    added here, exactly once, by ``PriceOutput`` to the state and to every
    price attribute (#39).
    With ``include_known`` the ``predictions`` attribute starts at the
    current slot with the confirmed spot prices, then the forecast (#40).
    """

    _attr_has_entity_name = True
    _attr_device_class = SensorDeviceClass.MONETARY
    _attr_icon = "mdi:brain"
    # The forecast series is live data for dashboards. The recorder drops
    # every attribute of a state whose attributes exceed 16 KB, so the
    # series is not recorded and the other attributes are (#103)
    _unrecorded_attributes = frozenset({"predictions"})

    def __init__(
        self,
        hass: HomeAssistant,
        entry: ConfigEntry,
        api_data: dict,
        currency: str,
        output: PriceOutput,
        prediction_hours: int = DEFAULT_PREDICTION_HOURS,
        attribute_format: str = DEFAULT_ATTRIBUTE_FORMAT,
        include_known: bool = DEFAULT_INCLUDE_KNOWN_PRICES,
    ):
        self.hass = hass
        self.entry = entry
        self.api_data = api_data
        self.currency = currency
        self.output = output
        self.compact = attribute_format == ATTRIBUTE_FORMAT_COMPACT
        self.include_known = include_known

        # Cap the predictions exposed as attributes to the configured window:
        # up to 72 hours in the detailed format, up to 168 in the compact
        # one (#38). The recorder does not store them (#103)
        hours = int(prediction_hours)
        if not self.compact:
            hours = min(hours, DETAILED_MAX_PREDICTION_HOURS)
        self._max_predictions = hours * HOUR_MINUTES // output.interval_minutes

        self._attr_unique_id = util_slugify(f"{DOMAIN}_{entry.entry_id}_ml_prediction")
        self._attr_name = "Price Forecast (ML)"
        self._attr_native_unit_of_measurement = output.unit(currency)
        self._attr_suggested_display_precision = output.precision

        self._attr_device_info = {
            "identifiers": {(DOMAIN, entry.entry_id)},
        }

    async def async_added_to_hass(self) -> None:
        async_dispatcher_connect(
            self.hass, util_slugify(UPDATE_SIGNAL_FORECAST), self._handle_update
        )

    async def _handle_update(self) -> None:
        self.async_write_ha_state()

    def _forecast(self) -> list[dict[str, Any]]:
        """Return the whole forecast as exposed (see ``PriceOutput.forecast``)."""
        ml_predictor = self.api_data.get("ml_predictor")
        if not ml_predictor or not ml_predictor.predictions:
            return []
        return self.output.forecast(ml_predictor.predictions, self._tariffs)

    @property
    def _tariffs(self) -> TariffSchedule | None:
        """Return the tariff schedule, None without one (#107)."""
        tariffs: TariffSchedule | None = self.api_data.get("tariffs")
        return tariffs

    @property
    def native_value(self) -> float | None:
        """Return the forecast price of the current slot (or hour).

        Falls back to the first future slot when the predictions start later.
        """
        entry = current_prediction(self._forecast(), dt_util.utcnow())
        return entry["price"] if entry is not None else None

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        ml_predictor = self.api_data.get("ml_predictor")
        attrs: dict[str, Any] = {}
        if ml_predictor:
            now = dt_util.utcnow()
            forecast = self._forecast()
            unit = self.output.unit(self.currency)
            spot_data = self.api_data.get("spot_data")
            series = (
                self.output.forecast(
                    with_known_prices(spot_data, ml_predictor.predictions, now),
                    self._tariffs,
                )
                if self.include_known
                else forecast
            )
            # The configured window (the detailed format is capped at 72 hours)
            window = series[: self._max_predictions]
            attrs["predictions"] = (
                compact_forecast(window, unit, self.output.interval_minutes)
                if self.compact
                else detailed_forecast(window, unit)
            )
            # The slot whose prediction is the state
            state_entry = current_prediction(forecast, now)
            attrs["state_slot_start"] = state_entry["start"] if state_entry else None
            # End of the confirmed spot prices: predictions start there (#40)
            known_end = known_until(spot_data)
            attrs["known_until"] = (
                dt_util.as_local(known_end).isoformat() if known_end else None
            )
            # Every price above and below: (spot + tariff + surcharge) + VAT
            attrs["includes_vat"] = True
            attrs["includes_tariffs"] = bool(self._tariffs)
            attrs["vat"] = self.output.vat
            attrs["surcharge"] = self.output.surcharge
            attrs["hourly_average"] = self.output.hourly_average
            stats = ml_predictor.get_prediction_stats()
            if stats:
                # Over the whole forecast, per slot (or hour), converted once
                prices = [
                    entry["price"]
                    for entry in self.output.forecast_series(
                        ml_predictor.predictions, self._tariffs
                    )
                ]
                if prices:
                    attrs["forecast_min"] = self.output.convert(min(prices))
                    attrs["forecast_max"] = self.output.convert(max(prices))
                    attrs["forecast_mean"] = self.output.convert(
                        sum(prices) / len(prices)
                    )
                attrs["unit"] = unit
                attrs["mean_confidence"] = stats.get("mean_confidence")
                attrs["total_predictions"] = stats.get("total_predictions")
                attrs["is_ml_model"] = stats.get("is_ml_model")
                attrs["training_samples"] = stats.get("training_samples")
            if self.compact:
                fit_compact(attrs, "predictions")
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
    # The per-slot metrics (96 slots) are live data too large to record
    # with the rest (#103)
    _unrecorded_attributes = frozenset({"hourly_metrics"})

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
            attrs["bias_offsets"] = metrics.get("bias_offsets")
            attrs["pending_predictions"] = metrics.get("pending_predictions")
            attrs["hourly_metrics"] = metrics.get("hourly_metrics")
            # Holdout error of the latest training (None before one)
            attrs["holdout_mae"] = metrics.get("holdout_mae")
            attrs["holdout_rmse"] = metrics.get("holdout_rmse")
            attrs["holdout_trained_at"] = metrics.get("holdout_trained_at")
        return attrs
