"""The update cycle: price reads, the forecast pipeline and the timed updates.

``async_setup_entry`` builds one ``ForecastUpdater`` per config entry, runs
its initial fetch and registers its callbacks: ``new_quarter`` every 15
minutes, ``update_forecasts`` every 6 hours, ``new_day`` at midnight and
``check_tomorrow_prices`` through ``TomorrowPriceChecker``. The forecast
pipeline (weather → Nordpool prognoses → known-data end → predict → save)
exists once, in ``run_forecast``.

Prices come from Stromligning's sensors or, with the day-ahead source
(#27), from energy-charts / ENTSO-E through ``DayAheadPrices``. Stored
history is kept by gap-aware sources (``api/time_series_source.py``, #32):
the forecast refreshes today's and tomorrow's prognoses, and the history the
model trains on is backfilled and pruned by ``HistoryUpdaterMixin``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.dispatcher import async_dispatcher_send
from homeassistant.util import dt as dt_util, slugify as util_slugify

from .api import NordpoolPrognosisSource, forecast_prognoses
from .const import (
    CONF_SOLAR_FORECAST_SENSOR,
    CONF_SOLAR_POWER_SENSOR,
    CONF_SPOT_PRICE_SENSOR,
    CONF_SPOT_PRICE_TOMORROW_SENSOR,
    CONF_STROMLIGNING_SENSOR,
    CONF_STROMLIGNING_TOMORROW_SENSOR,
    CONF_TEMPERATURE_SENSOR,
    CONF_WIND_DIRECTION_SENSOR,
    CONF_WIND_SPEED_SENSOR,
    DEFAULT_SPOT_PRICE_SENSOR,
    DEFAULT_SPOT_PRICE_TOMORROW_SENSOR,
    PRICE_SOURCE_DAYAHEAD,
    UPDATE_SIGNAL,
    UPDATE_SIGNAL_FORECAST,
)
from .history_updater import HistoryUpdaterMixin
from .ml.predictor import SpotPricePredictor
from .ml.storage import LearningStorage
from .price_source import DayAheadPrices, PriceSettings, with_vat
from .sensor_reader import SensorReader, async_read_weather_forecast
from .spot_prices import ml_price_inputs
from .time_slots import (
    floor_to_slot,
    local_midnight,
    slot_index_in_day,
    tomorrow_prices_complete,
    utc_slot_key,
)

_LOGGER = logging.getLogger(__name__)

# How far ahead the model predicts, and its slot length
FORECAST_DAYS = 7
INTERVAL_MINUTES = 15


@dataclass(frozen=True, slots=True)
class SensorEntities:
    """The external entities a config entry reads.

    Options (reconfiguration) take precedence over the entry's initial data.
    """

    stromligning: str | None
    stromligning_tomorrow: str | None
    # Raw spot price excl. VAT and tariffs: what the ML model learns (#16)
    spot_price: str | None
    spot_price_tomorrow: str | None
    wind_speed: str | None
    wind_direction: str | None
    solar_power: str | None
    solar_forecast: str | None
    temperature: str | None

    @classmethod
    def from_entry(cls, entry: ConfigEntry) -> SensorEntities:
        """Read the configured entity ids from a config entry."""

        def option(key: str, default: str | None = None) -> str | None:
            value: str | None = entry.options.get(key, entry.data.get(key, default))
            return value

        return cls(
            stromligning=option(CONF_STROMLIGNING_SENSOR),
            stromligning_tomorrow=option(CONF_STROMLIGNING_TOMORROW_SENSOR),
            spot_price=option(CONF_SPOT_PRICE_SENSOR, DEFAULT_SPOT_PRICE_SENSOR),
            spot_price_tomorrow=option(
                CONF_SPOT_PRICE_TOMORROW_SENSOR, DEFAULT_SPOT_PRICE_TOMORROW_SENSOR
            ),
            wind_speed=option(CONF_WIND_SPEED_SENSOR),
            wind_direction=option(CONF_WIND_DIRECTION_SENSOR),
            solar_power=option(CONF_SOLAR_POWER_SENSOR),
            solar_forecast=option(CONF_SOLAR_FORECAST_SENSOR),
            temperature=option(CONF_TEMPERATURE_SENSOR),
        )

    def sensor_config(self) -> dict[str, str | None]:
        """Return the ``api_data["sensor_config"]`` dict the weather reader uses."""
        return {
            "stromligning_sensor": self.stromligning,
            "wind_speed_sensor": self.wind_speed,
            "wind_direction_sensor": self.wind_direction,
            "solar_power_sensor": self.solar_power,
            "solar_forecast_sensor": self.solar_forecast,
            "temperature_sensor": self.temperature,
        }

    @property
    def has_weather(self) -> bool:
        """Return whether any weather or solar entity is configured."""
        return any(
            [
                self.wind_speed,
                self.wind_direction,
                self.solar_power,
                self.solar_forecast,
                self.temperature,
            ]
        )


class ForecastUpdater(HistoryUpdaterMixin):
    """Keeps a config entry's ``api_data`` current and runs the forecast."""

    def __init__(
        self,
        hass: HomeAssistant,
        entry: ConfigEntry,
        api_data: dict[str, Any],
        sensors: SensorEntities,
        sensor_reader: SensorReader,
        ml_predictor: SpotPricePredictor | None,
        settings: PriceSettings | None = None,
        storage: LearningStorage | None = None,
    ) -> None:
        """Initialize the updater; nothing is read until a method is called.

        Args:
            hass: Home Assistant instance.
            entry: The config entry being updated.
            api_data: The entry's shared data (``hass.data[DOMAIN][entry_id]``).
            sensors: The entities to read.
            sensor_reader: Reads every external entity.
            ml_predictor: The price model, or None with ML prediction disabled.
            settings: The price source; Stromligning's sensors if None.
            storage: The learning database the day-ahead prices are stored
                in (the model's, or a separate one without ML).
        """
        self.hass = hass
        self.entry = entry
        self.api_data = api_data
        self.sensors = sensors
        self.sensor_reader = sensor_reader
        self.ml_predictor = ml_predictor
        self.region: str = api_data["region"]
        # Only the ML model uses the prognoses, and they live in its database
        self.nordpool = (
            NordpoolPrognosisSource(hass, ml_predictor.storage, self.region)
            if ml_predictor
            else None
        )
        self.settings = settings or PriceSettings.from_entry(entry)
        self.dayahead = (
            DayAheadPrices(hass, storage, self.region, self.settings)
            if self.settings.dayahead and storage is not None
            else None
        )

    def _notify(self, signal: str) -> None:
        """Tell the entities that ``api_data`` changed."""
        async_dispatcher_send(self.hass, util_slugify(signal))

    # ------------------------------------------------------------------
    # Prices
    # ------------------------------------------------------------------

    def read_spot_prices(self) -> None:
        """Read the raw spot price (the ML model's prices) into api_data.

        With the day-ahead source they are read with the prices
        (``async_read_prices``).
        """
        if self.ml_predictor and self.dayahead is None:
            self.api_data["spot_data"] = self.sensor_reader.read_spot_prices(
                self.sensors.spot_price, self.sensors.spot_price_tomorrow
            )

    def read_prices(self) -> bool:
        """Read today's and tomorrow's prices into api_data.

        Reads Stromligning's consumer prices (displayed) and the raw spot
        prices (the model's).

        Returns:
            Whether Stromligning had valid prices for today. Without them the
            previous prices are kept: after a restart the sensor can be
            missing or report invalid (all-zero) prices until its source
            recovers.
        """
        _LOGGER.debug("Reading today's and tomorrow's prices")
        loaded = False
        if self.sensors.stromligning:
            stromligning_data = self.sensor_reader.read_stromligning_sensor(
                self.sensors.stromligning
            )
            _LOGGER.debug("Stromligning data: %s", stromligning_data)

            # Read tomorrow sensor if configured
            if self.sensors.stromligning_tomorrow:
                tomorrow_data = self.sensor_reader.read_stromligning_tomorrow_sensor(
                    self.sensors.stromligning_tomorrow
                )
                if tomorrow_data["available"] and tomorrow_data["tomorrow"]:
                    stromligning_data["tomorrow"] = tomorrow_data["tomorrow"]
                    stromligning_data["raw_tomorrow"] = tomorrow_data["raw_tomorrow"]
                    _LOGGER.info(
                        "Loaded %d tomorrow prices from Stromligning tomorrow sensor",
                        len(tomorrow_data["tomorrow"]),
                    )

            if stromligning_data["today"]:
                self.api_data["prices_today"] = stromligning_data["today"]
                self.api_data["prices_tomorrow"] = stromligning_data["tomorrow"]
                self.api_data["stromligning_data"] = stromligning_data
                self.api_data["price_source"] = "stromligning"
                loaded = True

        self.read_spot_prices()
        return loaded

    async def async_read_prices(self) -> bool:
        """Read today's and tomorrow's prices from the configured source.

        The day-ahead source fetches what is missing, converts the prices
        into the configured currency and shows them with VAT; its raw spot
        prices are the model's. Without today's prices the previous ones are
        kept.

        Returns:
            Whether today's prices were read.
        """
        if self.dayahead is None:
            return self.read_prices()
        try:
            spot_data = await self.dayahead.async_read(dt_util.now().date())
        except Exception as err:
            _LOGGER.warning("Could not read the day-ahead prices: %s", err)
            return False
        if not spot_data["today"]:
            return False
        vat = self.settings.vat
        self.api_data["spot_data"] = spot_data
        self.api_data["prices_today"] = with_vat(spot_data["today"], vat)
        self.api_data["prices_tomorrow"] = with_vat(spot_data["tomorrow"], vat)
        self.api_data["price_source"] = PRICE_SOURCE_DAYAHEAD
        return True

    def update_tomorrow_available(self) -> bool:
        """Recompute tomorrow_available; return True if tomorrow just became complete."""
        was_complete = self.api_data["tomorrow_available"]
        self.api_data["tomorrow_available"] = tomorrow_prices_complete(
            self.api_data["prices_tomorrow"]
        )
        arrived = bool(self.api_data["tomorrow_available"] and not was_complete)
        if arrived:
            _LOGGER.info(
                "Tomorrow's prices are complete (%d intervals)",
                len(self.api_data["prices_tomorrow"]),
            )
        return arrived

    # ------------------------------------------------------------------
    # Forecast pipeline
    # ------------------------------------------------------------------

    async def _read_weather(self) -> dict[str, Any]:
        """Read the weather sensors, Solcast and the hourly weather forecast."""
        weather_data: dict[str, Any] = {}
        solar_forecast = self.sensors.solar_forecast
        if self.sensors.has_weather:
            weather_data = self.sensor_reader.read_weather_sensors(
                self.api_data["sensor_config"]
            )
            _LOGGER.debug("Weather data from sensors: %s", weather_data)

            if solar_forecast and "solcast" in solar_forecast.lower():
                weather_data["solcast_forecast"] = (
                    self.sensor_reader.read_solcast_sensor(solar_forecast)
                )

        # Hourly weather forecast for time-varying per-slot features
        wind_speed = self.sensors.wind_speed
        if wind_speed and wind_speed.startswith("weather."):
            forecast = await async_read_weather_forecast(self.hass, wind_speed)
            if forecast:
                weather_data["weather_forecast"] = forecast
        return weather_data

    async def run_forecast(self) -> None:
        """Run the forecast pipeline once.

        Reads the weather and today's and tomorrow's Nordpool prognoses
        (refreshed from Nordpool), then predicts from the known raw
        spot prices onwards and saves the learning data. Without an ML
        predictor only the weather is read.
        """
        weather_data = await self._read_weather()
        self.api_data["weather_data"] = weather_data
        await self._update_prognoses(weather_data)

        ml_predictor = self.ml_predictor
        if ml_predictor and weather_data:
            # The model's prices: raw spot excl. VAT, today then tomorrow, and
            # where they end (predictions start there)
            all_known_prices, known_data_end_time = ml_price_inputs(
                self.api_data.get("spot_data")
            )
            _LOGGER.info(
                "Running ML predictions with %d price samples (known data ends at %s)",
                len(all_known_prices),
                known_data_end_time,
            )
            await self.hass.async_add_executor_job(
                ml_predictor.predict,
                weather_data,
                all_known_prices,
                FORECAST_DAYS,
                INTERVAL_MINUTES,
                known_data_end_time,
            )
            self.api_data["ml_predictions"] = ml_predictor.predictions
            _LOGGER.info("Generated %d ML predictions", len(ml_predictor.predictions))

            # Save learning data after prediction (includes stored predictions)
            await ml_predictor.save_learning_data()

    async def _update_prognoses(self, weather_data: dict[str, Any]) -> None:
        """Refresh today's and tomorrow's prognoses; attach the stored ones.

        Nordpool revises the current days, so they are re-fetched (an
        unpublished tomorrow waits for its retry time); a failed request
        keeps what is stored.
        """
        if self.nordpool is None:
            return
        today = dt_util.now().date()
        start = local_midnight(today)
        end = local_midnight(today + timedelta(days=2))
        try:
            await self.nordpool.async_update(start, end)
        except Exception as err:
            _LOGGER.warning("Could not update the Nordpool prognoses: %s", err)
        try:
            rows = await self.nordpool.async_load(start, end)
        except Exception as err:
            _LOGGER.warning("Could not read the stored Nordpool prognoses: %s", err)
            return
        weather_data.update(forecast_prognoses(rows))

    async def async_initial_fetch(self) -> None:
        """Read the prices and weather and run the first forecast at setup."""
        _LOGGER.info("Starting initial data fetch for Open Spot Forecast")
        api_data = self.api_data

        if self.dayahead is not None:
            if not await self.async_read_prices():
                _LOGGER.warning("No day-ahead prices for today yet")
        elif self.sensors.stromligning:
            _LOGGER.info(
                "Reading prices from Stromligning sensor: %s",
                self.sensors.stromligning,
            )
            if not self.read_prices():
                _LOGGER.warning("Stromligning sensor has no valid prices yet")
        else:
            self.read_prices()
            _LOGGER.warning("No price sensor configured (Stromligning recommended)")

        _LOGGER.info(
            "Price data loaded: source=%s, today=%d prices, tomorrow=%d prices",
            api_data.get("price_source", "unknown"),
            len(api_data["prices_today"]),
            len(api_data["prices_tomorrow"]),
        )
        self.update_tomorrow_available()

        if (
            self.ml_predictor
            and self.dayahead is None
            and not api_data["spot_data"]["today"]
        ):
            _LOGGER.warning(
                "Spot price sensor %s has no prices: the ML forecast needs the raw "
                "spot price excl. VAT (Stromligning's spotprice_ex_vat sensor)",
                self.sensors.spot_price,
            )

        await self.run_forecast()
        weather_data = api_data["weather_data"]
        _LOGGER.info(
            "Weather data loaded: wind_speed=%s, temperature=%s, solar_power=%s",
            weather_data.get("wind_speed"),
            weather_data.get("temperature"),
            weather_data.get("solar_power"),
        )

        api_data["last_update"] = datetime.now()
        self.start_history_backfill()
        _LOGGER.info("Initial data fetch completed successfully")

    # ------------------------------------------------------------------
    # Tomorrow's prices
    # ------------------------------------------------------------------

    async def refresh_forecast(self) -> None:
        """Re-read the prices and re-run the forecast (the model retrains on new data)."""
        await self.async_read_prices()
        self.update_tomorrow_available()
        self.api_data["last_update"] = datetime.now()
        if self.ml_predictor:
            await self.run_forecast()
        self._notify(UPDATE_SIGNAL)

    def start_forecast_refresh(self) -> None:
        """Refresh the forecast in the background once tomorrow is complete."""
        self.entry.async_create_background_task(
            self.hass, self.refresh_forecast(), "open_spot_forecast_tomorrow_prices"
        )

    async def check_tomorrow_prices(self) -> bool:
        """Re-read the prices; refresh the forecast when tomorrow completes.

        Called by TomorrowPriceChecker from 13:00 local until tomorrow's
        prices are complete. Returns whether they are.
        """
        await self.async_read_prices()
        self.api_data["last_update"] = datetime.now()
        if self.update_tomorrow_available() and self.ml_predictor:
            self.start_forecast_refresh()
        self._notify(UPDATE_SIGNAL)
        return bool(self.api_data["tomorrow_available"])

    # ------------------------------------------------------------------
    # Timed updates
    # ------------------------------------------------------------------

    async def update_forecasts(self, _now: datetime) -> None:
        """Update ML forecasts (every 6 hours)."""
        _LOGGER.info("6-hour forecast update triggered")
        if self.ml_predictor:
            await self.run_forecast()
        self._notify(UPDATE_SIGNAL_FORECAST)
        _LOGGER.debug("6-hour forecast update completed")

    async def new_day(self, _now: datetime) -> None:
        """Handle new day - rotate tomorrow to today."""
        _LOGGER.debug("New day - rotating prices")
        api_data = self.api_data

        if self.dayahead is not None:
            # Today's prices are stored since yesterday; tomorrow's come later
            api_data["prices_tomorrow"] = []
            await self.async_read_prices()
        elif self.sensors.stromligning:
            stromligning_data = self.sensor_reader.read_stromligning_sensor(
                self.sensors.stromligning
            )
            if stromligning_data["tomorrow"]:
                api_data["prices_today"] = stromligning_data["tomorrow"]
                api_data["prices_tomorrow"] = []
                api_data["stromligning_data"] = stromligning_data
            else:
                # No tomorrow data yet — clear it until the tomorrow-price check finds it
                api_data["prices_tomorrow"] = []
        else:
            api_data["prices_tomorrow"] = []
        api_data["tomorrow_available"] = False

        self.read_spot_prices()
        self._notify(UPDATE_SIGNAL)

        # Keep the stored history to the training window, and fill it in
        await self.prune_history()
        self.start_history_backfill()

    async def new_quarter(self, _now: datetime) -> None:
        """Update every 15 minutes and perform self-learning."""
        _LOGGER.info("15-minute update triggered for self-learning")
        api_data = self.api_data

        # Read the prices (displayed; also the tomorrow check). The day-ahead
        # source only requests missing prices, when they are due
        if self.dayahead is not None:
            await self.async_read_prices()
        elif self.sensors.stromligning:
            stromligning_data = self.sensor_reader.read_stromligning_sensor(
                self.sensors.stromligning
            )
            current_prices = stromligning_data["today"]
            if current_prices:
                api_data["prices_today"] = current_prices
                api_data["price_source"] = "stromligning"

            # Also check if tomorrow's prices are newly available
            if self.sensors.stromligning_tomorrow:
                tomorrow_data = self.sensor_reader.read_stromligning_tomorrow_sensor(
                    self.sensors.stromligning_tomorrow
                )
                if tomorrow_data["available"] and tomorrow_data["tomorrow"]:
                    api_data["prices_tomorrow"] = tomorrow_data["tomorrow"]

            _LOGGER.debug(
                "Read %d consumer prices from Stromligning", len(current_prices)
            )

        tomorrow_arrived = self.update_tomorrow_available()

        # The model learns from the raw spot price, never the consumer price
        self.read_spot_prices()
        spot_today: list[float | None] = (api_data.get("spot_data") or {}).get(
            "today", []
        )

        if self.ml_predictor and self.sensors.wind_speed:
            await self._store_weather_snapshot(self.ml_predictor)
        if self.ml_predictor and spot_today:
            await self._learn_current_slot(self.ml_predictor, spot_today)

        # Tomorrow's prices extend the training data: refresh the forecast now
        # (the model retrains on them) instead of waiting for the next run
        if tomorrow_arrived and self.ml_predictor:
            self.start_forecast_refresh()

        self._notify(UPDATE_SIGNAL)
        _LOGGER.debug("15-minute update completed, sensors notified")

    async def _store_weather_snapshot(self, ml_predictor: SpotPricePredictor) -> None:
        """Store the current weather for historical training."""
        try:
            # Keyed by the UTC slot start, the one stored timestamp format
            now_ts = utc_slot_key(dt_util.utcnow())
            weather_now = self.sensor_reader.read_weather_sensors(
                self.api_data["sensor_config"]
            )
            await self.hass.async_add_executor_job(
                ml_predictor.storage.insert_weather_snapshot,
                now_ts,
                weather_now.get("temperature"),
                weather_now.get("wind_speed"),
                weather_now.get("wind_direction"),
                weather_now.get("cloud_coverage"),
                weather_now.get("humidity"),
                weather_now.get("solar_power"),
            )
        except Exception as err:
            _LOGGER.debug("Failed to store weather snapshot: %s", err)

    async def _learn_current_slot(
        self, ml_predictor: SpotPricePredictor, spot_today: list[float | None]
    ) -> None:
        """Compare the stored predictions for the current slot with its price."""
        try:
            # Today's confirmed prices include the current slot, so match
            # every stored prediction for it, whatever its lead time. The
            # lookup date must be today's: it is paired with today's price.
            slot_time = dt_util.now()

            # Today's spot prices are one value per 15-minute slot from local
            # midnight (the reader aligns them by timestamp), so the
            # current slot's price is at its position on that grid
            learn_dt = floor_to_slot(slot_time)
            price_index = slot_index_in_day(slot_time)
            actual_price = (
                spot_today[price_index] if price_index < len(spot_today) else None
            )

            # A missing slot (a gap in the source) is not learned from
            if actual_price is None:
                return
            learn_timestamp = learn_dt.isoformat()

            _LOGGER.info(
                "Self-learning: looking up prediction for %s (index %d, price %.2f)",
                learn_timestamp,
                price_index,
                actual_price,
            )

            # Feed actual price to learning loop
            learning_did_update = await self.hass.async_add_executor_job(
                ml_predictor.learn_from_actual_price,
                learn_timestamp,
                actual_price,
            )

            if learning_did_update:
                _LOGGER.debug(
                    "Self-learning: compared prediction for %s with actual price %.2f",
                    learn_timestamp,
                    actual_price,
                )
                # Only persist if learning actually found a match
                await ml_predictor.save_learning_data()
        except Exception as err:
            _LOGGER.error("Self-learning update error: %s", err, exc_info=True)
