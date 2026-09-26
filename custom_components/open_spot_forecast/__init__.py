"""Open Spot Forecast integration for Home Assistant.

Sets up a config entry: reads its configuration, builds the ML predictor and
the ``ForecastUpdater`` (``updater.py``), runs the initial fetch and
registers the timed updates, which ``async_unload_entry`` cancels.
"""

import logging

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady
from homeassistant.helpers.event import async_track_time_change
from homeassistant.loader import async_get_integration

from .const import (
    CONF_CURRENCY,
    CONF_ENABLE_ML_PREDICTION,
    CONF_REGION,
    CONF_TRAINING_DAYS,
    DEFAULT_TRAINING_DAYS,
    DOMAIN,
    PLATFORMS,
    REGIONS,
    STARTUP,
)
from .ml.predictor import SpotPricePredictor
from .ml.storage import LearningStorage
from .price_source import PriceSettings
from .sensor_reader import SensorReader
from .tomorrow_prices import TomorrowPriceChecker
from .updater import ForecastUpdater, SensorEntities

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Open Spot Forecast from a config entry."""
    hass.data.setdefault(DOMAIN, {})
    integration = await async_get_integration(hass, DOMAIN)
    _LOGGER.info(STARTUP, integration.version)

    region = entry.data.get(CONF_REGION, "DK1")
    currency = entry.data.get(CONF_CURRENCY, "DKK")
    enable_ml = entry.options.get(
        CONF_ENABLE_ML_PREDICTION, entry.data.get(CONF_ENABLE_ML_PREDICTION, True)
    )

    # Sensor configuration - options first (reconfiguration), then data (initial setup)
    sensors = SensorEntities.from_entry(entry)
    price_settings = PriceSettings.from_entry(entry)
    _LOGGER.info(
        "Price source: %s (ENTSO-E fallback: %s)",
        price_settings.source,
        "configured" if price_settings.entsoe_api_key else "not configured",
    )
    _LOGGER.info(
        "Sensor configuration: stromligning=%s, stromligning_tomorrow=%s, spot=%s, "
        "spot_tomorrow=%s, wind_speed=%s, wind_direction=%s, solar_power=%s, "
        "solar_forecast=%s, temperature=%s",
        sensors.stromligning,
        sensors.stromligning_tomorrow,
        sensors.spot_price,
        sensors.spot_price_tomorrow,
        sensors.wind_speed,
        sensors.wind_direction,
        sensors.solar_power,
        sensors.solar_forecast,
        sensors.temperature,
    )

    # Initialize sensor reader
    sensor_reader = SensorReader(hass)

    # Initialize ML predictor
    ml_predictor = None
    if enable_ml:
        tz_name = str(REGIONS.get(region, {}).get("tz", "Europe/Copenhagen"))
        training_days = int(
            entry.options.get(
                CONF_TRAINING_DAYS,
                entry.data.get(CONF_TRAINING_DAYS, DEFAULT_TRAINING_DAYS),
            )
        )
        ml_predictor = SpotPricePredictor(hass, region, tz_name, training_days)
        # Load learning data asynchronously
        await ml_predictor._load_learning_data()
        await hass.async_add_executor_job(ml_predictor.refresh_lead_time_accuracy)

    # The day-ahead prices are stored in the learning database, which exists
    # without the ML model too
    storage: LearningStorage | None = ml_predictor.storage if ml_predictor else None
    own_storage = None
    if storage is None and price_settings.dayahead:
        own_storage = storage = await hass.async_add_executor_job(
            LearningStorage, hass, region
        )

    # Store API data
    api_data = {
        "ml_predictor": ml_predictor,
        "sensor_reader": sensor_reader,
        "region": region,
        "currency": currency,
        "prices_today": [],
        "prices_tomorrow": [],
        "ml_predictions": [],
        "weather_data": {},
        "tomorrow_available": False,
        "last_update": None,
        "sensor_config": sensors.sensor_config(),
        # Closed at unload: the day-ahead prices' database without ML
        "price_storage": own_storage,
    }

    hass.data[DOMAIN][entry.entry_id] = api_data

    updater = ForecastUpdater(
        hass,
        entry,
        api_data,
        sensors,
        sensor_reader,
        ml_predictor,
        price_settings,
        storage,
    )
    try:
        await updater.async_initial_fetch()
    except Exception as err:
        _LOGGER.error("Failed to initialize data: %s", err, exc_info=True)
        raise ConfigEntryNotReady from err

    # Schedule callbacks
    listeners = []

    # Tomorrow's prices: re-checked from 13:00 local until complete. The
    # checker reschedules itself, so unload cancels its pending check
    tomorrow_checker = TomorrowPriceChecker(hass, updater.check_tomorrow_prices)
    tomorrow_checker.schedule(api_data["tomorrow_available"])
    listeners.append(tomorrow_checker.cancel)

    # Forecasts every 6 hours
    if enable_ml:
        listeners.append(
            async_track_time_change(
                hass, updater.update_forecasts, hour="/6", minute=10, second=0
            )
        )

    # New day at midnight
    listeners.append(
        async_track_time_change(hass, updater.new_day, hour=0, minute=0, second=1)
    )

    # Every 15 minutes
    listeners.append(
        async_track_time_change(hass, updater.new_quarter, minute="/15", second=1)
    )

    api_data["listeners"] = listeners

    # Forward setup to platforms
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)

    if unload_ok:
        api_data = hass.data[DOMAIN].pop(entry.entry_id)
        for unsub in api_data.get("listeners", []):
            unsub()

        # Close persistent SQLite connections
        ml_predictor = api_data.get("ml_predictor")
        if ml_predictor is not None:
            await hass.async_add_executor_job(ml_predictor.storage.close)
        price_storage = api_data.get("price_storage")
        if price_storage is not None:
            await hass.async_add_executor_job(price_storage.close)

    return unload_ok


async def async_reload_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Reload config entry."""
    await async_unload_entry(hass, entry)
    await async_setup_entry(hass, entry)
