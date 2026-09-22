"""Open Spot Forecast integration for Home Assistant."""

import logging
from datetime import datetime, timedelta
from random import randint

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady
from homeassistant.helpers.dispatcher import async_dispatcher_send
from homeassistant.helpers.event import async_track_time_change
from homeassistant.loader import async_get_integration
from homeassistant.util import slugify as util_slugify

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

_LOGGER = logging.getLogger(__name__)


def _extract_latest_known_timestamp(
    raw_data_list: list, interval_minutes: int = 15
) -> datetime | None:
    """Find the end time of the latest known price from raw sensor data.

    Returns the timestamp after the last known interval, i.e. the point
    from which we should start predicting.
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
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            elif isinstance(ts, datetime):
                dt = ts
            else:
                continue
            if dt.tzinfo is not None:
                dt = dt.replace(tzinfo=None)
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

    Caching strategy:
    - Nordpool day-ahead data is published once daily (~13:00 CET) and
      never changes after publication.
    - Today's data: fetched once per day (validated via cache date).
    - Tomorrow's data: fetched once after 13:00 (when published).
    - If api_data is provided, cached data is reused and no API call
      is made when data is already fresh. This avoids redundant
      fetches in the 6-hourly update cycle.

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

    # --- Cache check ---
    if api_data is not None:
        cache = api_data.get("_nordpool_cache", {})
        cache_date = cache.get("date")  # ISO date string of last successful fetch

        # Today data is fresh if cached today (it never changes after publication)
        fresh_today = cache_date == today.isoformat() and cache.get("has_today")
        # Tomorrow data is fresh if cached after 13:00 (published time) on any
        # recent day. The data is static once published — it doesn't matter if
        # the cache was set today or yesterday, as long as it was post-13:00.
        cached_post_13 = cache.get("cached_after_13", False)
        fresh_tomorrow = cached_post_13 and cache.get("has_tomorrow")

        if fresh_today and fresh_tomorrow:
            # Inject cached data — no API call needed
            cached_cons = cache.get("consumption_prognosis")
            cached_prod = cache.get("production_prognosis")
            if cached_cons:
                weather_data["consumption_prognosis"] = cached_cons
            if cached_prod:
                weather_data["production_prognosis"] = cached_prod
            _LOGGER.debug(
                "Nordpool data reused from cache (date=%s, has_today=%s, has_tomorrow=%s)",
                cache_date,
                fresh_today,
                fresh_tomorrow,
            )
            return []

    # --- Fetch needed dates ---
    stored_entries: list[dict] = []
    # Always fetch today (most important — used immediately for prediction)
    dates_to_fetch = [today]
    # Tomorrow's day-ahead is published ~13:00 CET — only fetch if we're past
    # that OR at startup (api_data is None, meaning first load). Before 13:00
    # the API may return empty data for tomorrow, so skip it to avoid noise.
    if now.hour >= 13 or api_data is None:
        dates_to_fetch.append(tomorrow)

    for target in dates_to_fetch:
        consumption = await fetch_consumption_prognosis(target, region)
        production = await fetch_production_prognosis(target, region)

        if consumption:
            if "consumption_prognosis" not in weather_data:
                weather_data["consumption_prognosis"] = {}
            weather_data["consumption_prognosis"].update(consumption)

            # Build storage entries combining consumption + production
            for ts, cons in consumption.items():
                entry: dict = {
                    "timestamp": ts,
                    "consumption": cons,
                    "solar": None,
                    "wind_offshore": None,
                    "wind_onshore": None,
                }
                stored_entries.append(entry)

        if production:
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

    # --- Update cache ---
    if api_data is not None:
        has_cons = bool(weather_data.get("consumption_prognosis"))
        has_prod = bool(weather_data.get("production_prognosis"))
        api_data["_nordpool_cache"] = {
            "date": today.isoformat(),
            "has_today": today in dates_to_fetch and (has_cons or has_prod),
            "has_tomorrow": tomorrow in dates_to_fetch and (has_cons or has_prod),
            "cached_after_13": now.hour >= 13,
            "consumption_prognosis": weather_data.get(
                "consumption_prognosis", {}
            ).copy(),
            "production_prognosis": [
                dict(p) for p in weather_data.get("production_prognosis", [])
            ],
        }
        _LOGGER.info(
            "Nordpool cache updated: date=%s, has_today=%s, has_tomorrow=%s, after_13=%s",
            today.isoformat(),
            api_data["_nordpool_cache"]["has_today"],
            api_data["_nordpool_cache"]["has_tomorrow"],
            now.hour >= 13,
        )

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
                _LOGGER.warning("Stromligning sensor has no data")
                stromligning_sensor = None

        # No price sources — just use empty data
        if not stromligning_sensor:
            _LOGGER.warning("No price sensor configured (Stromligning recommended)")

        _LOGGER.info(
            "Price data loaded: source=%s, today=%d prices, tomorrow=%d prices",
            api_data.get("price_source", "unknown"),
            len(api_data["prices_today"]),
            len(api_data["prices_tomorrow"]),
        )

        # Set tomorrow availability flag (used by binary sensor)
        api_data["tomorrow_available"] = len(api_data["prices_tomorrow"]) >= 23

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
    rand_min = randint(10, 40)
    rand_sec = randint(0, 59)

    async def update_tomorrow_prices(_now):
        """Fetch tomorrow's prices (published ~13:00 CET)."""
        _LOGGER.debug("Fetching tomorrow's prices")

        # Try to read from Stromligning sensor first (priority 1)
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
            else:
                # Fall through to Nordpool/API
                pass

        # No price sensors configured
        if not stromligning_sensor:
            pass  # No fallback

        api_data["tomorrow_available"] = len(api_data["prices_tomorrow"]) >= 23
        api_data["last_update"] = datetime.now()

        # Update ML predictions if enabled
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
                # No tomorrow data yet — clear it until update_tomorrow_prices runs
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
                    if not api_data["prices_tomorrow"]:
                        _LOGGER.info(
                            "Tomorrow's prices now available (%d intervals)",
                            len(tomorrow_data["tomorrow"]),
                        )
                    api_data["prices_tomorrow"] = tomorrow_data["tomorrow"]

            _LOGGER.debug(
                "Read %d prices from Stromligning for learning", len(current_prices)
            )
        else:
            current_prices = api_data["prices_today"]
            _LOGGER.debug("Using %d cached prices for learning", len(current_prices))

        api_data["tomorrow_available"] = len(api_data["prices_tomorrow"]) >= 23

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
                # Learn from recent actual prices
                # We look at prices from 24 hours ago (predictions made yesterday)
                now = datetime.now()
                yesterday = now - timedelta(hours=24)

                # Determine the interval: 15-min Stromligning data has
                # 4 entries per hour; hourly Nordpool data has 1
                intervals_per_hour = 4 if len(current_prices) > 24 else 1

                # Build the timestamp at the current interval boundary
                learn_minute = (yesterday.minute // (60 // intervals_per_hour)) * (
                    60 // intervals_per_hour
                )
                learn_dt = yesterday.replace(
                    minute=learn_minute, second=0, microsecond=0
                )

                # Calculate the correct index into current_prices
                learn_hour = yesterday.hour
                price_index = learn_hour * intervals_per_hour + (
                    learn_minute // (60 // intervals_per_hour)
                )

                # Get actual price for that interval
                if len(current_prices) > price_index:
                    actual_price = current_prices[price_index]

                    # Format timestamp with timezone to match prediction format
                    if ml_predictor.tz:
                        learn_dt = learn_dt.replace(tzinfo=ml_predictor.tz)
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

        async_dispatcher_send(hass, util_slugify(UPDATE_SIGNAL))
        _LOGGER.debug("15-minute update completed, sensors notified")

    # Schedule callbacks
    listeners = []

    # Tomorrow's prices at ~13:xx CET
    listeners.append(
        async_track_time_change(
            hass, update_tomorrow_prices, hour=13, minute=rand_min, second=rand_sec
        )
    )

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
