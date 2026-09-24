"""Open Spot Forecast integration for Home Assistant."""

import logging
from datetime import datetime, timedelta

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady
from homeassistant.helpers.dispatcher import async_dispatcher_send
from homeassistant.helpers.event import async_track_time_change
from homeassistant.loader import async_get_integration
from homeassistant.util import dt as dt_util, slugify as util_slugify

from .api import fetch_consumption_prognosis, fetch_production_prognosis
from .const import (
    CONF_CURRENCY,
    CONF_ENABLE_ML_PREDICTION,
    CONF_REGION,
    CONF_SOLAR_FORECAST_SENSOR,
    CONF_SOLAR_POWER_SENSOR,
    CONF_STROMLIGNING_SENSOR,
    CONF_STROMLIGNING_TOMORROW_SENSOR,
    CONF_TEMPERATURE_SENSOR,
    CONF_WIND_DIRECTION_SENSOR,
    CONF_WIND_SPEED_SENSOR,
    DOMAIN,
    PLATFORMS,
    REGIONS,
    STARTUP,
    UPDATE_SIGNAL,
    UPDATE_SIGNAL_FORECAST,
)
from .ml.predictor import SpotPricePredictor
from .sensor_reader import SensorReader, async_read_weather_forecast
from .time_slots import floor_to_slot, slot_index_in_day, tomorrow_prices_complete
from .tomorrow_prices import TomorrowPriceChecker

_LOGGER = logging.getLogger(__name__)


def _extract_latest_known_timestamp(
    raw_data_list: list, interval_minutes: int = 15
) -> datetime | None:
    """Find the end time of the latest known price from raw sensor data.

    Returns the timestamp after the last known interval (UTC-aware), i.e.
    the point from which we should start predicting.
    """
    if not raw_data_list:
        return None

    latest = None
    for item in raw_data_list:
        if not isinstance(item, dict):
            continue
        ts = (
            item.get("timestamp")
            or item.get("time")
            or item.get("start")
            or item.get("end")
        )
        if ts is None:
            continue
        try:
            if isinstance(ts, str):
                dt = dt_util.parse_datetime(ts)
            elif isinstance(ts, datetime):
                dt = ts
            else:
                continue
            if dt is None:
                continue
            # Normalize to UTC (naive timestamps are assumed to be in HA's
            # local time zone) so comparisons are consistent.
            dt = dt_util.as_utc(dt)
            if latest is None or dt > latest:
                latest = dt
        except ValueError, TypeError:
            continue

    if latest is not None:
        # Return the start of the next interval after known data
        return latest + timedelta(minutes=interval_minutes)
    return None


async def _fetch_nordpool_prognoses(
    hass: HomeAssistant, region: str, weather_data: dict, api_data: dict | None = None
) -> list[dict]:
    """Fetch Nordpool consumption and production prognoses (with caching).

    Attaches 'consumption_prognosis' and 'production_prognosis' to
    weather_data if the API calls succeed. Also returns a list of
    entries suitable for storage in SQLite.

    Freshness is keyed on Nordpool's ``updatedAt`` timestamp rather than the
    wall-clock date. Day-ahead data is immutable once published, but the
    production per-type breakdown (Solar/WindOffshore/WindOnshore) is
    published later than the total, which surfaces as a later ``updatedAt``.
    We always fetch (calls are infrequent — startup + 6-hourly) and compare
    ``updatedAt`` so we pick up that late breakdown without guessing from the
    hour of day.

    Args:
        hass: Home Assistant instance
        region: Price region (e.g. "DK1")
        weather_data: Dict to attach fetched data to
        api_data: Optional integration data dict for caching

    Returns:
        List of dicts with keys: timestamp, consumption, solar,
        wind_offshore, wind_onshore
    """
    from datetime import date

    today = date.today()
    tomorrow = today + timedelta(days=1)
    now = datetime.now()

    cache = api_data.get("_nordpool_cache", {}) if api_data is not None else {}
    consumption_cache = cache.setdefault("consumption", {})
    production_cache = cache.setdefault("production", {})

    # Always fetch today. Tomorrow's day-ahead total is published ~13:00 CET,
    # but the per-type breakdown arrives later; fetching before 13:00 only
    # returns empty data, so skip it then.
    dates_to_fetch = [today]
    if now.hour >= 13 or api_data is None:
        dates_to_fetch.append(tomorrow)

    stored_entries: list[dict] = []
    for target in dates_to_fetch:
        date_key = target.isoformat()

        consumption, cons_updated = await fetch_consumption_prognosis(target, region)
        production, prod_updated = await fetch_production_prognosis(target, region)

        # --- Consumption ---
        if consumption:
            cached = consumption_cache.get(date_key)
            if cached and cons_updated == cached.get("updated_at"):
                # Unchanged since the last fetch — reuse what we already parsed.
                consumption = cached.get("data") or {}
            else:
                consumption_cache[date_key] = {
                    "updated_at": cons_updated,
                    "data": consumption,
                }

            if "consumption_prognosis" not in weather_data:
                weather_data["consumption_prognosis"] = {}
            weather_data["consumption_prognosis"].update(consumption)

            # Build storage entries combining consumption + production
            for ts, cons in consumption.items():
                stored_entries.append(
                    {
                        "timestamp": ts,
                        "consumption": cons,
                        "solar": None,
                        "wind_offshore": None,
                        "wind_onshore": None,
                    }
                )

        # --- Production ---
        if production:
            cached = production_cache.get(date_key)
            if cached and prod_updated == cached.get("updated_at"):
                production = cached.get("data") or []
            else:
                production_cache[date_key] = {
                    "updated_at": prod_updated,
                    "data": production,
                }

            if "production_prognosis" not in weather_data:
                weather_data["production_prognosis"] = []
            weather_data["production_prognosis"].extend(production)

            # Update storage entries with production data
            prod_by_ts = {p["deliveryStart"]: p for p in production}
            for entry in stored_entries:
                ts = entry["timestamp"]
                if ts in prod_by_ts:
                    p = prod_by_ts[ts]
                    entry["solar"] = p.get("solar")
                    entry["wind_offshore"] = p.get("wind_offshore")
                    entry["wind_onshore"] = p.get("wind_onshore")

    if api_data is not None:
        api_data["_nordpool_cache"] = cache
        _LOGGER.info("Nordpool cache updated for %d dates", len(dates_to_fetch))

    return stored_entries


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

    # Sensor configuration - check options first (reconfiguration), then data (initial setup)
    stromligning_sensor = entry.options.get(
        CONF_STROMLIGNING_SENSOR, entry.data.get(CONF_STROMLIGNING_SENSOR)
    )
    stromligning_tomorrow_sensor = entry.options.get(
        CONF_STROMLIGNING_TOMORROW_SENSOR,
        entry.data.get(CONF_STROMLIGNING_TOMORROW_SENSOR),
    )
    wind_speed_sensor = entry.options.get(
        CONF_WIND_SPEED_SENSOR, entry.data.get(CONF_WIND_SPEED_SENSOR)
    )
    wind_direction_sensor = entry.options.get(
        CONF_WIND_DIRECTION_SENSOR, entry.data.get(CONF_WIND_DIRECTION_SENSOR)
    )
    solar_power_sensor = entry.options.get(
        CONF_SOLAR_POWER_SENSOR, entry.data.get(CONF_SOLAR_POWER_SENSOR)
    )
    solar_forecast_sensor = entry.options.get(
        CONF_SOLAR_FORECAST_SENSOR, entry.data.get(CONF_SOLAR_FORECAST_SENSOR)
    )
    temperature_sensor = entry.options.get(
        CONF_TEMPERATURE_SENSOR, entry.data.get(CONF_TEMPERATURE_SENSOR)
    )

    _LOGGER.info(
        "Sensor configuration: stromligning=%s, stromligning_tomorrow=%s, wind_speed=%s, "
        "wind_direction=%s, solar_power=%s, solar_forecast=%s, temperature=%s",
        stromligning_sensor,
        stromligning_tomorrow_sensor,
        wind_speed_sensor,
        wind_direction_sensor,
        solar_power_sensor,
        solar_forecast_sensor,
        temperature_sensor,
    )

    # Initialize sensor reader
    sensor_reader = SensorReader(hass)

    # Initialize ML predictor
    ml_predictor = None
    if enable_ml:
        tz_name = str(REGIONS.get(region, {}).get("tz", "Europe/Copenhagen"))
        ml_predictor = SpotPricePredictor(hass, region, tz_name)
        # Load learning data asynchronously
        await ml_predictor._load_learning_data()
        await hass.async_add_executor_job(ml_predictor.refresh_lead_time_accuracy)

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
        "sensor_config": {
            "stromligning_sensor": stromligning_sensor,
            "wind_speed_sensor": wind_speed_sensor,
            "wind_direction_sensor": wind_direction_sensor,
            "solar_power_sensor": solar_power_sensor,
            "solar_forecast_sensor": solar_forecast_sensor,
            "temperature_sensor": temperature_sensor,
        },
    }

    hass.data[DOMAIN][entry.entry_id] = api_data

    def update_tomorrow_available() -> bool:
        """Recompute tomorrow_available; return True if tomorrow just became complete."""
        was_complete = api_data["tomorrow_available"]
        api_data["tomorrow_available"] = tomorrow_prices_complete(
            api_data["prices_tomorrow"]
        )
        arrived = api_data["tomorrow_available"] and not was_complete
        if arrived:
            _LOGGER.info(
                "Tomorrow's prices are complete (%d intervals)",
                len(api_data["prices_tomorrow"]),
            )
        return arrived

    # Initial data fetch
    try:
        _LOGGER.info("Starting initial data fetch for Open Spot Forecast")

        # Try to read from Stromligning sensor first (priority 1)
        if stromligning_sensor:
            _LOGGER.info(
                "Reading prices from Stromligning sensor: %s", stromligning_sensor
            )
            stromligning_data = sensor_reader.read_stromligning_sensor(
                stromligning_sensor
            )
            _LOGGER.debug("Stromligning data: %s", stromligning_data)

            # Read tomorrow sensor if configured
            if stromligning_tomorrow_sensor:
                _LOGGER.info(
                    "Reading tomorrow prices from Stromligning sensor: %s",
                    stromligning_tomorrow_sensor,
                )
                tomorrow_data = sensor_reader.read_stromligning_tomorrow_sensor(
                    stromligning_tomorrow_sensor
                )
                if tomorrow_data["available"] and tomorrow_data["tomorrow"]:
                    stromligning_data["tomorrow"] = tomorrow_data["tomorrow"]
                    stromligning_data["raw_tomorrow"] = tomorrow_data["raw_tomorrow"]
                    _LOGGER.info(
                        "Loaded %d tomorrow prices from Stromligning tomorrow sensor",
                        len(tomorrow_data["tomorrow"]),
                    )

            if stromligning_data["today"]:
                api_data["prices_today"] = stromligning_data["today"]
                api_data["prices_tomorrow"] = stromligning_data["tomorrow"]
                api_data["stromligning_data"] = stromligning_data
                api_data["price_source"] = "stromligning"
                _LOGGER.info(
                    "Loaded %d today prices and %d tomorrow prices from Stromligning (real consumer prices with tariffs/VAT)",
                    len(stromligning_data["today"]),
                    len(stromligning_data["tomorrow"]),
                )
            else:
                # Keep reading the sensor: after a restart it can be missing or
                # report invalid (all-zero) prices until its source recovers
                _LOGGER.warning("Stromligning sensor has no valid prices yet")

        # No price sources — just use empty data
        if not stromligning_sensor:
            _LOGGER.warning("No price sensor configured (Stromligning recommended)")

        _LOGGER.info(
            "Price data loaded: source=%s, today=%d prices, tomorrow=%d prices",
            api_data.get("price_source", "unknown"),
            len(api_data["prices_today"]),
            len(api_data["prices_tomorrow"]),
        )

        # Set tomorrow availability flag
        update_tomorrow_available()

        # Read weather data from sensors or DMI API
        weather_data = {}
        if any(
            [
                wind_speed_sensor,
                wind_direction_sensor,
                solar_power_sensor,
                solar_forecast_sensor,
                temperature_sensor,
            ]
        ):
            _LOGGER.info("Reading weather data from Home Assistant sensors")
            _LOGGER.debug("Sensor config: %s", api_data["sensor_config"])
            weather_data = sensor_reader.read_weather_sensors(api_data["sensor_config"])
            _LOGGER.debug("Weather data from sensors: %s", weather_data)

            # Also read Solcast if configured
            if solar_forecast_sensor and "solcast" in solar_forecast_sensor.lower():
                solcast_data = sensor_reader.read_solcast_sensor(solar_forecast_sensor)
                weather_data["solcast_forecast"] = solcast_data
                _LOGGER.debug("Solcast data: %s", solcast_data)

        api_data["weather_data"] = weather_data

        # Read hourly weather forecast for time-varying per-slot features
        if wind_speed_sensor and wind_speed_sensor.startswith("weather."):
            forecast = await async_read_weather_forecast(hass, wind_speed_sensor)
            if forecast:
                weather_data["weather_forecast"] = forecast
                api_data["weather_data"] = weather_data

        # Fetch Nordpool consumption and production prognoses
        np_entries = await _fetch_nordpool_prognoses(
            hass, region, weather_data, api_data
        )
        api_data["weather_data"] = weather_data
        if np_entries and ml_predictor:
            await hass.async_add_executor_job(
                ml_predictor.storage.insert_nordpool_prognoses_batch, np_entries
            )

        _LOGGER.info(
            "Weather data loaded: wind_speed=%s, temperature=%s, solar_power=%s",
            weather_data.get("wind_speed"),
            weather_data.get("temperature"),
            weather_data.get("solar_power"),
        )

        if ml_predictor and weather_data:
            # Combine today and tomorrow prices for historical data
            all_known_prices = list(api_data["prices_today"])
            if api_data["prices_tomorrow"]:
                all_known_prices.extend(api_data["prices_tomorrow"])

            # Determine where known data ends (so we don't predict confirmed prices)
            known_data_end_time = None
            price_source_data = api_data.get("stromligning_data")
            if price_source_data:
                raw_combined = list(price_source_data.get("raw_today", [])) + list(
                    price_source_data.get("raw_tomorrow", [])
                )
                known_data_end_time = _extract_latest_known_timestamp(raw_combined)

            _LOGGER.info(
                "Running ML predictions with %d price samples (known data ends at %s)",
                len(all_known_prices),
                known_data_end_time,
            )
            await hass.async_add_executor_job(
                ml_predictor.predict,
                weather_data,
                all_known_prices,
                7,  # forecast_days
                15,  # interval_minutes
                known_data_end_time,
            )
            api_data["ml_predictions"] = ml_predictor.predictions
            _LOGGER.info("Generated %d ML predictions", len(ml_predictor.predictions))

            # Save learning data after prediction (includes stored predictions)
            await ml_predictor.save_learning_data()

        api_data["last_update"] = datetime.now()
        _LOGGER.info("Initial data fetch completed successfully")

    except Exception as err:
        _LOGGER.error("Failed to initialize data: %s", err, exc_info=True)
        raise ConfigEntryNotReady from err

    # Schedule updates
    def read_stromligning_prices() -> None:
        """Read today's and tomorrow's prices from Stromligning into api_data."""
        _LOGGER.debug("Reading today's and tomorrow's prices")

        if stromligning_sensor:
            stromligning_data = sensor_reader.read_stromligning_sensor(
                stromligning_sensor
            )
            _LOGGER.debug("Stromligning data (update): %s", stromligning_data)

            # Read tomorrow sensor if configured
            if stromligning_tomorrow_sensor:
                tomorrow_data = sensor_reader.read_stromligning_tomorrow_sensor(
                    stromligning_tomorrow_sensor
                )
                if tomorrow_data["available"] and tomorrow_data["tomorrow"]:
                    stromligning_data["tomorrow"] = tomorrow_data["tomorrow"]
                    stromligning_data["raw_tomorrow"] = tomorrow_data["raw_tomorrow"]
                    _LOGGER.info(
                        "Loaded %d tomorrow prices from Stromligning tomorrow sensor",
                        len(tomorrow_data["tomorrow"]),
                    )

            if stromligning_data["today"]:
                api_data["prices_today"] = stromligning_data["today"]
                api_data["prices_tomorrow"] = stromligning_data["tomorrow"]
                api_data["stromligning_data"] = stromligning_data
                api_data["price_source"] = "stromligning"
                _LOGGER.debug("Updated prices from Stromligning")

    async def refresh_forecast() -> None:
        """Re-read the prices and re-run the forecast (the model retrains on new data)."""
        read_stromligning_prices()
        update_tomorrow_available()
        api_data["last_update"] = datetime.now()

        if ml_predictor:
            # Read weather data
            weather_data = {}
            if any(
                [
                    wind_speed_sensor,
                    wind_direction_sensor,
                    solar_power_sensor,
                    solar_forecast_sensor,
                    temperature_sensor,
                ]
            ):
                weather_data = sensor_reader.read_weather_sensors(
                    api_data["sensor_config"]
                )

                if solar_forecast_sensor and "solcast" in solar_forecast_sensor.lower():
                    solcast_data = sensor_reader.read_solcast_sensor(
                        solar_forecast_sensor
                    )
                    weather_data["solcast_forecast"] = solcast_data

            api_data["weather_data"] = weather_data

            # Read hourly weather forecast for time-varying per-slot features
            if wind_speed_sensor and wind_speed_sensor.startswith("weather."):
                forecast = await async_read_weather_forecast(hass, wind_speed_sensor)
                if forecast:
                    weather_data["weather_forecast"] = forecast
                    api_data["weather_data"] = weather_data

            # Fetch Nordpool consumption and production prognoses
            np_entries = await _fetch_nordpool_prognoses(
                hass, region, weather_data, api_data
            )
            api_data["weather_data"] = weather_data
            if np_entries and ml_predictor:
                await hass.async_add_executor_job(
                    ml_predictor.storage.insert_nordpool_prognoses_batch, np_entries
                )

            if weather_data:
                # Combine today and tomorrow prices for historical data
                all_known_prices = list(api_data["prices_today"])
                if api_data["prices_tomorrow"]:
                    all_known_prices.extend(api_data["prices_tomorrow"])

                # Determine where known data ends
                known_data_end_time = None
                price_source_data = api_data.get("stromligning_data")
                if price_source_data:
                    raw_combined = list(price_source_data.get("raw_today", [])) + list(
                        price_source_data.get("raw_tomorrow", [])
                    )
                    known_data_end_time = _extract_latest_known_timestamp(raw_combined)

                await hass.async_add_executor_job(
                    ml_predictor.predict,
                    weather_data,
                    all_known_prices,
                    7,  # forecast_days
                    15,  # interval_minutes
                    known_data_end_time,
                )
                api_data["ml_predictions"] = ml_predictor.predictions

                # Save learning data after prediction
                await ml_predictor.save_learning_data()

        async_dispatcher_send(hass, util_slugify(UPDATE_SIGNAL))

    def start_forecast_refresh() -> None:
        """Refresh the forecast in the background once tomorrow is complete."""
        entry.async_create_background_task(
            hass, refresh_forecast(), "open_spot_forecast_tomorrow_prices"
        )

    async def check_tomorrow_prices() -> bool:
        """Re-read the prices; refresh the forecast when tomorrow completes.

        Called by TomorrowPriceChecker from 13:00 local until tomorrow's
        prices are complete. Returns whether they are.
        """
        read_stromligning_prices()
        api_data["last_update"] = datetime.now()
        if update_tomorrow_available() and ml_predictor:
            start_forecast_refresh()
        async_dispatcher_send(hass, util_slugify(UPDATE_SIGNAL))
        return bool(api_data["tomorrow_available"])

    async def update_forecasts(_now):
        """Update ML forecasts (every 6 hours)."""
        _LOGGER.info("6-hour forecast update triggered")

        if ml_predictor:
            # Read weather data
            weather_data = {}
            if any(
                [
                    wind_speed_sensor,
                    wind_direction_sensor,
                    solar_power_sensor,
                    solar_forecast_sensor,
                    temperature_sensor,
                ]
            ):
                weather_data = sensor_reader.read_weather_sensors(
                    api_data["sensor_config"]
                )
                _LOGGER.debug("Weather data for forecast update: %s", weather_data)

                if solar_forecast_sensor and "solcast" in solar_forecast_sensor.lower():
                    solcast_data = sensor_reader.read_solcast_sensor(
                        solar_forecast_sensor
                    )
                    weather_data["solcast_forecast"] = solcast_data

            # Read hourly weather forecast for time-varying per-slot features
            if wind_speed_sensor and wind_speed_sensor.startswith("weather."):
                forecast = await async_read_weather_forecast(hass, wind_speed_sensor)
                if forecast:
                    weather_data["weather_forecast"] = forecast
                    api_data["weather_data"] = weather_data

            # Fetch Nordpool consumption and production prognoses (cached)
            np_entries = await _fetch_nordpool_prognoses(
                hass, region, weather_data, api_data
            )
            api_data["weather_data"] = weather_data
            if np_entries and ml_predictor:
                await hass.async_add_executor_job(
                    ml_predictor.storage.insert_nordpool_prognoses_batch, np_entries
                )

            if weather_data:
                _LOGGER.info("Running ML predictions with updated weather data")

                # Combine today and tomorrow prices for historical data
                all_known_prices = list(api_data["prices_today"])
                if api_data["prices_tomorrow"]:
                    all_known_prices.extend(api_data["prices_tomorrow"])

                # Determine where known data ends
                known_data_end_time = None
                price_source_data = api_data.get("stromligning_data")
                if price_source_data:
                    raw_combined = list(price_source_data.get("raw_today", [])) + list(
                        price_source_data.get("raw_tomorrow", [])
                    )
                    known_data_end_time = _extract_latest_known_timestamp(raw_combined)

                await hass.async_add_executor_job(
                    ml_predictor.predict,
                    weather_data,
                    all_known_prices,
                    7,  # forecast_days
                    15,  # interval_minutes
                    known_data_end_time,
                )
                api_data["ml_predictions"] = ml_predictor.predictions
                _LOGGER.info(
                    "Generated %d ML predictions", len(ml_predictor.predictions)
                )

                # Save learning data after prediction
                await ml_predictor.save_learning_data()

        async_dispatcher_send(hass, util_slugify(UPDATE_SIGNAL_FORECAST))
        _LOGGER.debug("6-hour forecast update completed")

    async def new_day(_now):
        """Handle new day - rotate tomorrow to today."""
        _LOGGER.debug("New day - rotating prices")

        # Try to read from Stromligning sensor first (priority 1)
        if stromligning_sensor:
            stromligning_data = sensor_reader.read_stromligning_sensor(
                stromligning_sensor
            )
            if stromligning_data["tomorrow"]:
                api_data["prices_today"] = stromligning_data["tomorrow"]
                api_data["prices_tomorrow"] = []
                api_data["stromligning_data"] = stromligning_data
            else:
                # No tomorrow data yet — clear it until the tomorrow-price check finds it
                api_data["prices_tomorrow"] = []
            api_data["tomorrow_available"] = False

        # No price sensors configured
        if not stromligning_sensor:
            api_data["prices_tomorrow"] = []
            api_data["tomorrow_available"] = False

        async_dispatcher_send(hass, util_slugify(UPDATE_SIGNAL))

    async def new_quarter(_now):
        """Update every 15 minutes and perform self-learning."""
        _LOGGER.info("15-minute update triggered for self-learning")

        # Read current prices once (used for both tomorrow check and learning)
        current_prices: list[float] = []
        if stromligning_sensor:
            stromligning_data = sensor_reader.read_stromligning_sensor(
                stromligning_sensor
            )
            current_prices = stromligning_data["today"]
            if current_prices:
                api_data["prices_today"] = current_prices
                api_data["price_source"] = "stromligning"

            # Also check if tomorrow's prices are newly available
            if stromligning_tomorrow_sensor:
                tomorrow_data = sensor_reader.read_stromligning_tomorrow_sensor(
                    stromligning_tomorrow_sensor
                )
                if tomorrow_data["available"] and tomorrow_data["tomorrow"]:
                    api_data["prices_tomorrow"] = tomorrow_data["tomorrow"]

            _LOGGER.debug(
                "Read %d prices from Stromligning for learning", len(current_prices)
            )
        else:
            current_prices = api_data["prices_today"]
            _LOGGER.debug("Using %d cached prices for learning", len(current_prices))

        tomorrow_arrived = update_tomorrow_available()

        # --- Collect weather snapshot for historical training ---
        if ml_predictor and wind_speed_sensor:
            try:
                now = datetime.now()
                now_ts = now.isoformat()
                weather_now = sensor_reader.read_weather_sensors(
                    api_data["sensor_config"]
                )
                await hass.async_add_executor_job(
                    ml_predictor.storage.insert_weather_snapshot,
                    now_ts,
                    weather_now.get("temperature"),
                    weather_now.get("wind_speed"),
                    weather_now.get("wind_direction"),
                    weather_now.get("cloud_coverage"),
                    weather_now.get("humidity"),
                    weather_now.get("solar_power"),
                )
                # Prune old weather every 100 snapshots
                if ml_predictor.storage.count_weather_snapshots() % 100 == 0:
                    ml_predictor.storage.delete_old_weather(30)
            except Exception as err:
                _LOGGER.debug("Failed to store weather snapshot: %s", err)

        # --- Self-learning: compare past predictions with actual prices ---
        if ml_predictor and current_prices:
            try:
                # Today's confirmed prices include the current slot, so match
                # every stored prediction for it, whatever its lead time. The
                # lookup date must be today's: it is paired with today's price.
                slot_time = dt_util.now()

                # Determine the interval: 15-min Stromligning data has 92-100
                # entries per day; hourly data has 23-25
                interval_minutes = 15 if len(current_prices) > 25 else 60

                # The current slot, and its position in today's prices counted
                # from local midnight on the UTC timeline, so a 92- or 100-slot
                # DST day is indexed correctly (not hour * 4 + minute // 15)
                learn_dt = floor_to_slot(slot_time, interval_minutes)
                price_index = slot_index_in_day(
                    slot_time, interval_minutes=interval_minutes
                )

                # Get actual price for that interval
                if len(current_prices) > price_index:
                    actual_price = current_prices[price_index]

                    learn_timestamp = learn_dt.isoformat()

                    _LOGGER.info(
                        "Self-learning: looking up prediction for %s (index %d, price %.2f)",
                        learn_timestamp,
                        price_index,
                        actual_price,
                    )

                    # Feed actual price to learning loop
                    learning_did_update = await hass.async_add_executor_job(
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

        # Tomorrow's prices extend the training data: refresh the forecast now
        # (the model retrains on them) instead of waiting for the next run
        if tomorrow_arrived and ml_predictor:
            start_forecast_refresh()

        async_dispatcher_send(hass, util_slugify(UPDATE_SIGNAL))
        _LOGGER.debug("15-minute update completed, sensors notified")

    # Schedule callbacks
    listeners = []

    # Tomorrow's prices: re-checked from 13:00 local until complete. The
    # checker reschedules itself, so unload cancels its pending check
    tomorrow_checker = TomorrowPriceChecker(hass, check_tomorrow_prices)
    tomorrow_checker.schedule(api_data["tomorrow_available"])
    listeners.append(tomorrow_checker.cancel)

    # Forecasts every 6 hours
    if enable_ml:
        listeners.append(
            async_track_time_change(
                hass, update_forecasts, hour="/6", minute=10, second=0
            )
        )

    # New day at midnight
    listeners.append(async_track_time_change(hass, new_day, hour=0, minute=0, second=1))

    # Every 15 minutes
    listeners.append(async_track_time_change(hass, new_quarter, minute="/15", second=1))

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

        # Close persistent SQLite connection
        ml_predictor = api_data.get("ml_predictor")
        if ml_predictor is not None:
            await hass.async_add_executor_job(ml_predictor.storage.close)

    return unload_ok


async def async_reload_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Reload config entry."""
    await async_unload_entry(hass, entry)
    await async_setup_entry(hass, entry)
