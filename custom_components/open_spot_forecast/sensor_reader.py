"""Sensor reader for Home Assistant integrations."""

import contextlib
import logging
from datetime import datetime
from typing import Any

from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)


class SensorReader:
    """Reads data from existing Home Assistant sensors."""

    def __init__(self, hass: HomeAssistant):
        """Initialize sensor reader."""
        self.hass = hass

    def get_sensor_state(self, entity_id: str) -> float | None:
        """Get current state of a sensor as float.

        Args:
            entity_id: Entity ID of the sensor

        Returns:
            Sensor state as float, or None if unavailable
        """
        if not entity_id:
            return None

        state = self.hass.states.get(entity_id)
        if state is None:
            _LOGGER.debug("Sensor %s not found", entity_id)
            return None

        if state.state in ("unknown", "unavailable", ""):
            _LOGGER.debug("Sensor %s is unavailable", entity_id)
            return None

        try:
            return float(state.state)
        except ValueError, TypeError:
            _LOGGER.debug("Sensor %s state '%s' is not numeric", entity_id, state.state)
            return None

    def get_sensor_attribute(self, entity_id: str, attribute: str) -> Any:
        """Get attribute from a sensor.

        Args:
            entity_id: Entity ID of the sensor
            attribute: Attribute name

        Returns:
            Attribute value, or None if not found
        """
        if not entity_id:
            return None

        state = self.hass.states.get(entity_id)
        if state is None:
            return None

        return state.attributes.get(attribute)

    def read_stromligning_sensor(self, entity_id: str) -> dict:
        """Read Stromligning sensor data.

        Tries multiple attribute names for price arrays:
        - prices (15-min intervals)
        - today/tomorrow (hourly)
        - raw_today/raw_tomorrow

        Falls back to using current price if no arrays found.
        """
        result = {
            "current_price": None,
            "today": [],
            "tomorrow": [],
            "raw_today": [],
            "raw_tomorrow": [],
            "prices_15min": [],
        }

        if not entity_id:
            return result

        state = self.hass.states.get(entity_id)
        if state is None:
            _LOGGER.warning("Stromligning sensor %s not found", entity_id)
            return result

        # Current price
        try:
            result["current_price"] = float(state.state)
        except ValueError, TypeError:
            _LOGGER.debug("Stromligning sensor state is not numeric")

        # Log all available attributes for debugging
        _LOGGER.debug(
            "Stromligning sensor attributes: %s", list(state.attributes.keys())
        )

        # Try to find price arrays in various attribute names
        prices_attr = None
        for attr_name in ["prices", "price_list", "forecast", "hourly_prices"]:
            if attr_name in state.attributes:
                prices_attr = state.attributes[attr_name]
                _LOGGER.debug("Found prices in attribute: %s", attr_name)
                break

        if prices_attr and isinstance(prices_attr, list):
            # Parse the price array
            from datetime import datetime, timedelta

            now = datetime.now()
            today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            tomorrow_start = today_start + timedelta(days=1)

            for item in prices_attr:
                if isinstance(item, dict):
                    price = item.get("price") or item.get("value")
                    timestamp = (
                        item.get("timestamp") or item.get("time") or item.get("start")
                    )

                    if price is not None and timestamp is not None:
                        try:
                            # Parse timestamp
                            if isinstance(timestamp, str):
                                dt = datetime.fromisoformat(
                                    timestamp.replace("Z", "+00:00")
                                )
                                # Remove timezone info for comparison
                                if dt.tzinfo is not None:
                                    dt = dt.replace(tzinfo=None)
                            else:
                                dt = timestamp
                                # Remove timezone info if present
                                if hasattr(dt, "tzinfo") and dt.tzinfo is not None:
                                    dt = dt.replace(tzinfo=None)

                            price_val = float(price)

                            # Categorize as today or tomorrow
                            if today_start <= dt < tomorrow_start:
                                result["today"].append(price_val)
                                result["raw_today"].append(item)
                            elif dt >= tomorrow_start:
                                result["tomorrow"].append(price_val)
                                result["raw_tomorrow"].append(item)

                            result["prices_15min"].append(item)
                        except (ValueError, TypeError) as e:
                            _LOGGER.debug("Error parsing price item: %s", e)

        # Try today/tomorrow attributes
        if not result["today"] and "today" in state.attributes:
            today_data = state.attributes["today"]
            if isinstance(today_data, list):
                result["today"] = [float(p) for p in today_data if p is not None]
                _LOGGER.debug("Found today prices in 'today' attribute")

        if not result["tomorrow"] and "tomorrow" in state.attributes:
            tomorrow_data = state.attributes["tomorrow"]
            if isinstance(tomorrow_data, list):
                result["tomorrow"] = [float(p) for p in tomorrow_data if p is not None]
                _LOGGER.debug("Found tomorrow prices in 'tomorrow' attribute")

        # If we still have no prices but have current price, use it as fallback.
        # This is expected during the midnight rollover when the sensor clears
        # its price arrays but still reports a current price, so log at debug.
        if not result["today"] and result["current_price"] is not None:
            _LOGGER.debug(
                "Stromligning sensor has current price but no price arrays. "
                "Using current price as fallback for today."
            )
            result["today"] = [result["current_price"]]

        _LOGGER.info(
            "Read Stromligning sensor %s: current=%.4f kr/kWh, "
            "today=%d intervals, tomorrow=%d intervals",
            entity_id,
            result["current_price"] or 0,
            len(result["today"]),
            len(result["tomorrow"]),
        )

        return result

    def read_stromligning_tomorrow_sensor(self, entity_id: str) -> dict:
        """Read Stromligning tomorrow sensor data.

        This sensor contains tomorrow's prices in the 'prices' attribute.
        """
        result = {
            "tomorrow": [],
            "raw_tomorrow": [],
            "available": False,
        }

        if not entity_id:
            return result

        state = self.hass.states.get(entity_id)
        if state is None:
            _LOGGER.debug("Stromligning tomorrow sensor %s not found", entity_id)
            return result

        # Check if tomorrow prices are available
        result["available"] = state.state == "on"

        if not result["available"]:
            _LOGGER.debug(
                "Stromligning tomorrow sensor is off, prices not yet available"
            )
            return result

        # Parse prices attribute
        prices_attr = state.attributes.get("prices", [])
        if not prices_attr or not isinstance(prices_attr, list):
            _LOGGER.debug("Stromligning tomorrow sensor has no prices attribute")
            return result

        from datetime import datetime

        for item in prices_attr:
            if isinstance(item, dict):
                price = item.get("price") or item.get("value")
                timestamp = (
                    item.get("timestamp") or item.get("time") or item.get("start")
                )

                if price is not None and timestamp is not None:
                    try:
                        # Parse timestamp
                        if isinstance(timestamp, str):
                            dt = datetime.fromisoformat(
                                timestamp.replace("Z", "+00:00")
                            )
                            # Remove timezone info for comparison
                            if dt.tzinfo is not None:
                                dt = dt.replace(tzinfo=None)
                        else:
                            dt = timestamp
                            # Remove timezone info if present
                            if hasattr(dt, "tzinfo") and dt.tzinfo is not None:
                                dt = dt.replace(tzinfo=None)

                        price_val = float(price)
                        result["tomorrow"].append(price_val)
                        result["raw_tomorrow"].append(item)
                    except (ValueError, TypeError) as e:
                        _LOGGER.debug("Error parsing tomorrow price item: %s", e)

        _LOGGER.info(
            "Read Stromligning tomorrow sensor %s: %d intervals",
            entity_id,
            len(result["tomorrow"]),
        )

        return result

    def read_weather_sensors(self, config: dict) -> dict:
        """Read weather data from configured sensors.

        Supports both sensor and weather entities. For weather entities
        (e.g. weather.forecast_mellemlokken_23), reads attributes like
        wind_speed, wind_bearing, temperature directly.

        Args:
            config: Configuration dictionary with sensor entity IDs

        Returns:
            Dictionary with weather data
        """
        weather_data = {
            "wind_speed": None,
            "wind_direction": None,
            "solar_power": None,
            "solar_forecast": None,
            "temperature": None,
            "cloud_coverage": None,
            "humidity": None,
            "timestamp": datetime.now().isoformat(),
        }

        # Wind speed - can come from weather entity attribute or sensor
        wind_speed_entity = config.get("wind_speed_sensor")
        if wind_speed_entity:
            if wind_speed_entity.startswith("weather."):
                state = self.hass.states.get(wind_speed_entity)
                if state:
                    wind_speed = state.attributes.get("wind_speed")
                    if wind_speed is not None:
                        unit = state.attributes.get("wind_speed_unit", "km/h")
                        if unit == "km/h":
                            weather_data["wind_speed"] = float(wind_speed) / 3.6
                        else:
                            weather_data["wind_speed"] = float(wind_speed)
            else:
                weather_data["wind_speed"] = self.get_sensor_state(wind_speed_entity)

        if weather_data["wind_speed"] is None:
            _LOGGER.warning(
                "Wind speed sensor %s not available (may load later), "
                "wind features will be zero until next forecast update",
                wind_speed_entity or "(none configured)",
            )

        # Wind direction - can come from weather entity attribute or sensor
        wind_dir_entity = config.get("wind_direction_sensor")
        if wind_dir_entity:
            if wind_dir_entity.startswith("weather."):
                state = self.hass.states.get(wind_dir_entity)
                if state:
                    wind_bearing = state.attributes.get("wind_bearing")
                    if wind_bearing is not None:
                        weather_data["wind_direction"] = float(wind_bearing)
            else:
                weather_data["wind_direction"] = self.get_sensor_state(wind_dir_entity)

        # Solar power (current production)
        solar_power_entity = config.get("solar_power_sensor")
        if solar_power_entity:
            weather_data["solar_power"] = self.get_sensor_state(solar_power_entity)

        # Solar forecast - read Solcast intervals/detailedHourly attributes
        solar_forecast_entity = config.get("solar_forecast_sensor")
        if solar_forecast_entity:
            solcast_data = self.read_solcast_sensor(solar_forecast_entity)
            weather_data["solar_forecast"] = solcast_data

        # Temperature - can come from weather entity state or sensor
        temp_entity = config.get("temperature_sensor")
        if temp_entity:
            if temp_entity.startswith("weather."):
                state = self.hass.states.get(temp_entity)
                if state:
                    with contextlib.suppress(ValueError, TypeError):
                        weather_data["temperature"] = float(state.state)
            else:
                weather_data["temperature"] = self.get_sensor_state(temp_entity)

        if weather_data["temperature"] is None:
            _LOGGER.warning(
                "Temperature sensor %s not available (may load later), "
                "using default 15.0°C until next forecast update",
                temp_entity or "(none configured)",
            )

        # Cloud coverage and humidity from weather entity
        if wind_speed_entity and wind_speed_entity.startswith("weather."):
            state = self.hass.states.get(wind_speed_entity)
            if state:
                cc = state.attributes.get("cloud_coverage")
                if cc is not None:
                    weather_data["cloud_coverage"] = float(cc)
                hum = state.attributes.get("humidity")
                if hum is not None:
                    weather_data["humidity"] = float(hum)

        _LOGGER.debug(
            "Read weather sensors: wind_speed=%s, wind_dir=%s, solar_power=%s, temp=%s",
            weather_data["wind_speed"],
            weather_data["wind_direction"],
            weather_data["solar_power"],
            weather_data["temperature"],
        )

        return weather_data

    def read_solcast_sensor(self, entity_id: str) -> dict:
        """Read Solcast solar forecast sensor.

        Solcast provides rich forecast data in attributes:
        - intervals: 30-minute forecast with spread_kwh and confidence
        - detailedHourly: hourly forecast with pv_estimate, pv_estimate10, pv_estimate90
        - estimate: today's total estimate in kWh

        Args:
            entity_id: Solcast sensor entity ID (e.g. sensor.solcast_pv_forecast_forecast_today)

        Returns:
            Dictionary with solar forecast data
        """
        result = {
            "current_power": None,
            "estimate_today": None,
            "estimate10": None,
            "estimate90": None,
            "intervals": [],
            "detailed_hourly": [],
        }

        if not entity_id:
            return result

        state = self.hass.states.get(entity_id)
        if state is None:
            _LOGGER.warning("Solcast sensor %s not found", entity_id)
            return result

        # Current power (state value)
        with contextlib.suppress(ValueError, TypeError):
            result["current_power"] = float(state.state)

        # Today's estimates
        result["estimate_today"] = state.attributes.get("estimate")
        result["estimate10"] = state.attributes.get("estimate10")
        result["estimate90"] = state.attributes.get("estimate90")

        # 30-minute intervals
        intervals = state.attributes.get("intervals", [])
        if intervals:
            result["intervals"] = intervals

        # Hourly detailed forecast
        detailed_hourly = state.attributes.get("detailedHourly", [])
        if detailed_hourly:
            result["detailed_hourly"] = detailed_hourly

        _LOGGER.debug(
            "Read Solcast sensor %s: current=%.2f kWh, estimate=%.2f kWh, %d intervals, %d hourly",
            entity_id,
            result["current_power"] or 0,
            result["estimate_today"] or 0,
            len(result["intervals"]),
            len(result["detailed_hourly"]),
        )

        return result

    def read_met_weather(self, entity_id: str) -> dict:
        """Read Met.no weather sensor.

        Met.no provides:
        - Temperature (state)
        - Wind speed, direction
        - Humidity, pressure
        - Forecast data in attributes

        Args:
            entity_id: Met.no weather entity ID

        Returns:
            Dictionary with weather data
        """
        result = {
            "temperature": None,
            "wind_speed": None,
            "wind_direction": None,
            "humidity": None,
            "pressure": None,
            "forecast": [],
        }

        if not entity_id:
            return result

        state = self.hass.states.get(entity_id)
        if state is None:
            _LOGGER.warning("Met.no sensor %s not found", entity_id)
            return result

        # Current conditions
        result["temperature"] = self.get_sensor_state(entity_id)
        result["wind_speed"] = state.attributes.get("wind_speed")
        result["wind_direction"] = state.attributes.get("wind_bearing")
        result["humidity"] = state.attributes.get("humidity")
        result["pressure"] = state.attributes.get("pressure")

        # Forecast data
        forecast = state.attributes.get("forecast", [])
        if forecast:
            result["forecast"] = forecast

        _LOGGER.debug(
            "Read Met.no sensor %s: temp=%s, wind_speed=%s, wind_dir=%s",
            entity_id,
            result["temperature"],
            result["wind_speed"],
            result["wind_direction"],
        )

        return result


async def async_read_weather_forecast(
    hass: HomeAssistant, entity_id: str
) -> list[dict] | None:
    """Read hourly weather forecast via weather.get_forecasts service.

    Returns a list of hourly forecast entries, each with:
        datetime, wind_speed, wind_bearing, temperature,
        cloud_coverage, humidity, precipitation

    Returns None if the service call fails or entity doesn't exist.
    """
    try:
        state = hass.states.get(entity_id)
        if state is None:
            _LOGGER.warning("Weather entity %s not found for forecast", entity_id)
            return None

        response = await hass.services.async_call(
            "weather",
            "get_forecasts",
            {"entity_id": entity_id, "type": "hourly"},
            blocking=True,
            return_response=True,
        )

        if not response or not isinstance(response, dict) or entity_id not in response:
            _LOGGER.debug("No hourly forecast available for %s", entity_id)
            return None

        forecast_data: dict = response[entity_id]  # type: ignore[assignment]
        if not isinstance(forecast_data, dict):
            _LOGGER.debug("Unexpected forecast data format for %s", entity_id)
            return None

        forecast_list = forecast_data.get("forecast", [])
        if not isinstance(forecast_list, list) or not forecast_list:
            _LOGGER.debug("Empty forecast list for %s", entity_id)
            return None

        _LOGGER.info(
            "Read %d hourly weather forecasts from %s",
            len(forecast_list),
            entity_id,
        )
        return forecast_list

    except Exception as err:
        _LOGGER.warning("Failed to read weather forecast for %s: %s", entity_id, err)
        return None
