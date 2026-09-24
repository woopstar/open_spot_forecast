"""Tests for the SensorReader module."""

import logging
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from homeassistant.util import dt as dt_util

from custom_components.open_spot_forecast.sensor_reader import (
    SensorReader,
    async_read_weather_forecast,
)


def _state(value: str, attributes: dict | None = None) -> MagicMock:
    """Build a mock Home Assistant state object."""
    state = MagicMock()
    state.state = value
    state.attributes = attributes if attributes is not None else {}
    return state


def _make_reader(states: dict | None = None) -> SensorReader:
    """Build a SensorReader backed by a mock state registry."""
    hass = Mock()
    mapping = states or {}
    hass.states.get.side_effect = lambda entity_id: mapping.get(entity_id)
    return SensorReader(hass)


class TestGetSensorState:
    """Tests for SensorReader.get_sensor_state."""

    def test_empty_entity_id(self):
        """An empty entity id returns None."""
        assert _make_reader().get_sensor_state("") is None

    def test_missing_entity(self):
        """A missing entity returns None."""
        assert _make_reader().get_sensor_state("sensor.missing") is None

    def test_unknown_state(self):
        """An 'unknown' state returns None."""
        reader = _make_reader({"sensor.x": _state("unknown")})
        assert reader.get_sensor_state("sensor.x") is None

    def test_unavailable_state(self):
        """An 'unavailable' state returns None."""
        reader = _make_reader({"sensor.x": _state("unavailable")})
        assert reader.get_sensor_state("sensor.x") is None

    def test_non_numeric_state(self):
        """A non-numeric state returns None."""
        reader = _make_reader({"sensor.x": _state("on")})
        assert reader.get_sensor_state("sensor.x") is None

    def test_numeric_state(self):
        """A numeric state is converted to a float."""
        reader = _make_reader({"sensor.x": _state("123.45")})
        assert reader.get_sensor_state("sensor.x") == 123.45


class TestGetSensorAttribute:
    """Tests for SensorReader.get_sensor_attribute."""

    def test_empty_entity_id(self):
        """An empty entity id returns None."""
        assert _make_reader().get_sensor_attribute("", "attr") is None

    def test_missing_entity(self):
        """A missing entity returns None."""
        assert _make_reader().get_sensor_attribute("sensor.missing", "attr") is None

    def test_present_attribute(self):
        """A present attribute value is returned."""
        reader = _make_reader({"sensor.x": _state("1.0", {"foo": "bar"})})
        assert reader.get_sensor_attribute("sensor.x", "foo") == "bar"

    def test_absent_attribute(self):
        """An absent attribute returns None."""
        reader = _make_reader({"sensor.x": _state("1.0", {})})
        assert reader.get_sensor_attribute("sensor.x", "missing") is None


class TestReadStromligningSensor:
    """Tests for SensorReader.read_stromligning_sensor."""

    def test_empty_entity_id(self):
        """An empty entity id returns the default empty result."""
        result = _make_reader().read_stromligning_sensor("")
        assert result["current_price"] is None
        assert result["today"] == []
        assert result["tomorrow"] == []
        assert result["raw_today"] == []
        assert result["raw_tomorrow"] == []
        assert result["prices_15min"] == []

    def test_missing_entity(self):
        """A missing entity returns the default empty result."""
        result = _make_reader().read_stromligning_sensor("sensor.missing")
        assert result["current_price"] is None
        assert result["today"] == []

    def test_non_numeric_state(self):
        """A non-numeric state leaves current_price as None."""
        reader = _make_reader({"sensor.strom": _state("on", {})})
        result = reader.read_stromligning_sensor("sensor.strom")
        assert result["current_price"] is None
        assert result["today"] == []

    def test_prices_attribute_full_parse(self):
        """A full prices array is parsed and categorized into today/tomorrow."""
        now = datetime.now()
        today_iso = now.isoformat()
        tomorrow_iso = (now + timedelta(days=1)).isoformat()
        aware_iso = now.replace(microsecond=0).astimezone().isoformat()
        aware_dt = now.astimezone()
        items = [
            {"price": 1.23, "timestamp": today_iso},
            {"value": 2.34, "time": tomorrow_iso},
            {"price": 3.45, "start": aware_iso},
            {"price": 4.56, "timestamp": now},
            {"price": 5.67, "timestamp": aware_dt},
            {"price": "bad", "timestamp": today_iso},
            {},
        ]
        reader = _make_reader({"sensor.strom": _state("9.99", {"prices": items})})

        result = reader.read_stromligning_sensor("sensor.strom")

        assert result["current_price"] == 9.99
        assert result["today"] == [1.23, 3.45, 4.56, 5.67]
        assert result["tomorrow"] == [2.34]
        assert len(result["raw_today"]) == 4
        assert len(result["raw_tomorrow"]) == 1
        assert len(result["prices_15min"]) == 5

    @pytest.mark.parametrize(
        "attr_name", ["prices", "price_list", "forecast", "hourly_prices"]
    )
    def test_price_array_attribute_names(self, attr_name):
        """Every supported price-array attribute name is recognised."""
        now = datetime.now()
        reader = _make_reader(
            {
                "sensor.strom": _state(
                    "5.0", {attr_name: [{"price": 1.0, "timestamp": now.isoformat()}]}
                )
            }
        )

        result = reader.read_stromligning_sensor("sensor.strom")

        assert result["today"] == [1.0]

    def test_today_tomorrow_attributes(self):
        """The 'today'/'tomorrow' attribute lists are used as a fallback."""
        reader = _make_reader(
            {
                "sensor.strom": _state(
                    "3.0",
                    {"today": [1.0, 2.0], "tomorrow": [4.0, 5.0]},
                )
            }
        )

        result = reader.read_stromligning_sensor("sensor.strom")

        assert result["today"] == [1.0, 2.0]
        assert result["tomorrow"] == [4.0, 5.0]

    def test_prices_not_a_list_falls_back(self):
        """A non-list prices attribute falls back to today/tomorrow lists."""
        reader = _make_reader(
            {
                "sensor.strom": _state(
                    "5.0",
                    {"prices": "not-a-list", "today": [1.0], "tomorrow": [2.0]},
                )
            }
        )

        result = reader.read_stromligning_sensor("sensor.strom")

        assert result["today"] == [1.0]
        assert result["tomorrow"] == [2.0]

    def test_midnight_rollover_fallback(self):
        """Current price is used as today's fallback when no arrays exist."""
        reader = _make_reader({"sensor.strom": _state("6.0", {})})

        result = reader.read_stromligning_sensor("sensor.strom")

        assert result["current_price"] == 6.0
        assert result["today"] == [6.0]


class TestReadStromligningTomorrowSensor:
    """Tests for SensorReader.read_stromligning_tomorrow_sensor."""

    def test_empty_entity_id(self):
        """An empty entity id returns the default result."""
        result = _make_reader().read_stromligning_tomorrow_sensor("")
        assert result == {"tomorrow": [], "raw_tomorrow": [], "available": False}

    def test_missing_entity(self):
        """A missing entity returns the default result."""
        result = _make_reader().read_stromligning_tomorrow_sensor("sensor.missing")
        assert result["available"] is False
        assert result["tomorrow"] == []

    def test_off_state(self):
        """An 'off' state means tomorrow prices are not yet available."""
        reader = _make_reader({"sensor.tomorrow": _state("off", {})})
        result = reader.read_stromligning_tomorrow_sensor("sensor.tomorrow")
        assert result["available"] is False
        assert result["tomorrow"] == []

    def test_on_state_no_prices(self):
        """An 'on' state with no prices attribute returns available but empty."""
        reader = _make_reader({"sensor.tomorrow": _state("on", {})})
        result = reader.read_stromligning_tomorrow_sensor("sensor.tomorrow")
        assert result["available"] is True
        assert result["tomorrow"] == []

    def test_on_state_prices_not_a_list(self):
        """A non-list prices attribute returns available but empty."""
        reader = _make_reader({"sensor.tomorrow": _state("on", {"prices": "x"})})
        result = reader.read_stromligning_tomorrow_sensor("sensor.tomorrow")
        assert result["available"] is True
        assert result["tomorrow"] == []

    def test_prices_parsed(self):
        """Prices are parsed into tomorrow and raw_tomorrow."""
        now = datetime.now()
        aware_iso = now.astimezone().isoformat()
        aware_dt = now.astimezone()
        items = [
            {"price": 1.0, "timestamp": (now + timedelta(days=1)).isoformat()},
            {"value": 2.0, "time": now.isoformat()},
            {"price": 3.0, "start": now},
            {"price": 4.0, "timestamp": aware_iso},
            {"price": 5.0, "timestamp": aware_dt},
            {"price": "bad", "timestamp": now.isoformat()},
            {},
        ]
        reader = _make_reader({"sensor.tomorrow": _state("on", {"prices": items})})

        result = reader.read_stromligning_tomorrow_sensor("sensor.tomorrow")

        assert result["available"] is True
        assert result["tomorrow"] == [1.0, 2.0, 3.0, 4.0, 5.0]
        assert len(result["raw_tomorrow"]) == 5


def _day_items(prices: list[float], day_offset: int = 0) -> list[dict]:
    """Return 15-minute price items for a local day (0 = today, 1 = tomorrow)."""
    start = dt_util.start_of_local_day() + timedelta(days=day_offset)
    return [
        {"price": price, "start": (start + timedelta(minutes=15 * i)).isoformat()}
        for i, price in enumerate(prices)
    ]


class TestRejectInvalidPriceDays:
    """All-zero or incomplete days are dropped at the source (issue #21)."""

    def test_all_zero_today_is_rejected_and_valid_tomorrow_kept(self, caplog):
        """An all-zero today reads as "no data"; tomorrow is judged on its own."""
        items = _day_items([0.0] * 96) + _day_items([0.5] * 96, day_offset=1)
        reader = _make_reader({"sensor.strom": _state("0.0", {"prices": items})})

        result = reader.read_stromligning_sensor("sensor.strom")

        assert result["today"] == []
        assert result["raw_today"] == []
        assert result["tomorrow"] == pytest.approx([0.5] * 96)
        assert len(result["raw_tomorrow"]) == 96
        assert "Ignoring today's prices from sensor.strom" in caplog.text

    def test_zero_and_negative_prices_are_kept(self):
        """Some zero or negative prices are valid, and a 0 price is not dropped."""
        prices = [0.0, -0.12, 0.35, 0.0]
        reader = _make_reader(
            {"sensor.strom": _state("0.35", {"prices": _day_items(prices)})}
        )

        result = reader.read_stromligning_sensor("sensor.strom")

        assert result["today"] == pytest.approx(prices)
        assert len(result["raw_today"]) == 4

    def test_rejection_warns_once_per_streak(self, caplog):
        """Repeated invalid reads warn once, and again after valid data."""
        zero = _state("0.0", {"prices": _day_items([0.0] * 96)})
        good = _state("1.0", {"prices": _day_items([1.0] * 96)})
        states = {"sensor.strom": zero}
        reader = _make_reader(states)

        with caplog.at_level(logging.DEBUG):
            reader.read_stromligning_sensor("sensor.strom")
            reader.read_stromligning_sensor("sensor.strom")
            states["sensor.strom"] = good
            reader.read_stromligning_sensor("sensor.strom")
            states["sensor.strom"] = zero
            reader.read_stromligning_sensor("sensor.strom")

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 2

    def test_missing_value_in_today_attribute_rejects_the_day(self):
        """A None slot invalidates the day instead of shifting later slots."""
        reader = _make_reader(
            {"sensor.strom": _state("3.0", {"today": [1.0, None, 2.0]})}
        )

        result = reader.read_stromligning_sensor("sensor.strom")

        assert result["today"] == []

    def test_zero_current_price_fallback_is_rejected(self):
        """The current-price fallback is rejected when that price is 0."""
        reader = _make_reader({"sensor.strom": _state("0", {})})

        result = reader.read_stromligning_sensor("sensor.strom")

        assert result["current_price"] == pytest.approx(0.0)
        assert result["today"] == []

    def test_all_zero_tomorrow_sensor_is_rejected(self):
        """The tomorrow sensor's all-zero prices read as "not published yet"."""
        items = _day_items([0.0] * 96, day_offset=1)
        reader = _make_reader({"sensor.tomorrow": _state("on", {"prices": items})})

        result = reader.read_stromligning_tomorrow_sensor("sensor.tomorrow")

        assert result["tomorrow"] == []
        assert result["raw_tomorrow"] == []


class TestReadWeatherSensors:
    """Tests for SensorReader.read_weather_sensors."""

    def test_nothing_configured(self):
        """An empty config yields all-None weather data."""
        data = _make_reader().read_weather_sensors({})
        assert data["wind_speed"] is None
        assert data["wind_direction"] is None
        assert data["solar_power"] is None
        assert data["solar_forecast"] is None
        assert data["temperature"] is None
        assert data["cloud_coverage"] is None
        assert data["humidity"] is None

    def test_plain_sensors(self):
        """Plain sensor entities are read via get_sensor_state."""
        states = {
            "sensor.wind": _state("12.5"),
            "sensor.wind_dir": _state("180"),
            "sensor.solar": _state("500"),
            "sensor.solcast": _state(
                "2.5",
                {
                    "estimate": 10.0,
                    "estimate10": 9.0,
                    "estimate90": 11.0,
                    "intervals": [{"period_start": "x"}],
                    "detailedHourly": [{"period_start": "y"}],
                },
            ),
            "sensor.temp": _state("18.5"),
        }
        config = {
            "wind_speed_sensor": "sensor.wind",
            "wind_direction_sensor": "sensor.wind_dir",
            "solar_power_sensor": "sensor.solar",
            "solar_forecast_sensor": "sensor.solcast",
            "temperature_sensor": "sensor.temp",
        }

        data = _make_reader(states).read_weather_sensors(config)

        assert data["wind_speed"] == 12.5
        assert data["wind_direction"] == 180.0
        assert data["solar_power"] == 500.0
        assert data["solar_forecast"]["estimate_today"] == 10.0
        assert data["temperature"] == 18.5

    def test_weather_entity_kmh(self):
        """A weather entity wind speed in km/h is converted to m/s."""
        attrs = {
            "wind_speed": 36.0,
            "wind_speed_unit": "km/h",
            "wind_bearing": 270,
            "cloud_coverage": 80,
            "humidity": 55,
        }
        state = _state("21.0", attrs)
        config = {
            "wind_speed_sensor": "weather.test",
            "wind_direction_sensor": "weather.test",
            "temperature_sensor": "weather.test",
        }

        data = _make_reader({"weather.test": state}).read_weather_sensors(config)

        assert data["wind_speed"] == 10.0
        assert data["wind_direction"] == 270.0
        assert data["temperature"] == 21.0
        assert data["cloud_coverage"] == 80.0
        assert data["humidity"] == 55.0

    def test_weather_entity_ms(self):
        """A weather entity wind speed already in m/s is kept as-is."""
        attrs = {"wind_speed": 5.0, "wind_speed_unit": "m/s"}
        data = _make_reader(
            {"weather.test": _state("20.0", attrs)}
        ).read_weather_sensors({"wind_speed_sensor": "weather.test"})
        assert data["wind_speed"] == 5.0

    def test_weather_entity_wind_attribute_missing(self):
        """A weather entity without a wind_speed attribute yields None."""
        data = _make_reader({"weather.test": _state("20.0", {})}).read_weather_sensors(
            {"wind_speed_sensor": "weather.test"}
        )
        assert data["wind_speed"] is None

    def test_weather_entity_missing(self):
        """A configured but missing weather entity yields None values."""
        config = {
            "wind_speed_sensor": "weather.missing",
            "temperature_sensor": "weather.missing",
        }
        data = _make_reader().read_weather_sensors(config)
        assert data["wind_speed"] is None
        assert data["temperature"] is None

    def test_weather_entity_non_numeric_temperature(self):
        """A non-numeric weather temperature state is suppressed to None."""
        data = _make_reader({"weather.test": _state("on", {})}).read_weather_sensors(
            {"temperature_sensor": "weather.test"}
        )
        assert data["temperature"] is None


class TestReadSolcastSensor:
    """Tests for SensorReader.read_solcast_sensor."""

    def test_empty_entity_id(self):
        """An empty entity id returns the default result."""
        result = _make_reader().read_solcast_sensor("")
        assert result == {
            "current_power": None,
            "estimate_today": None,
            "estimate10": None,
            "estimate90": None,
            "intervals": [],
            "detailed_hourly": [],
        }

    def test_missing_entity(self):
        """A missing entity returns the default result."""
        result = _make_reader().read_solcast_sensor("sensor.solcast")
        assert result["current_power"] is None
        assert result["intervals"] == []

    def test_full_attributes(self):
        """All Solcast attributes are surfaced."""
        intervals = [{"period_start": "x"}]
        detailed_hourly = [{"period_start": "y"}]
        state = _state(
            "2.5",
            {
                "estimate": 10.0,
                "estimate10": 9.0,
                "estimate90": 11.0,
                "intervals": intervals,
                "detailedHourly": detailed_hourly,
            },
        )

        result = _make_reader({"sensor.solcast": state}).read_solcast_sensor(
            "sensor.solcast"
        )

        assert result["current_power"] == 2.5
        assert result["estimate_today"] == 10.0
        assert result["estimate10"] == 9.0
        assert result["estimate90"] == 11.0
        assert result["intervals"] == intervals
        assert result["detailed_hourly"] == detailed_hourly

    def test_non_numeric_state_and_no_arrays(self):
        """A non-numeric state is suppressed and empty arrays stay empty."""
        state = _state("on", {"estimate": 1.0})

        result = _make_reader({"sensor.solcast": state}).read_solcast_sensor(
            "sensor.solcast"
        )

        assert result["current_power"] is None
        assert result["estimate_today"] == 1.0
        assert result["intervals"] == []
        assert result["detailed_hourly"] == []


class TestReadMetWeather:
    """Tests for SensorReader.read_met_weather."""

    def test_empty_entity_id(self):
        """An empty entity id returns the default result."""
        result = _make_reader().read_met_weather("")
        assert result == {
            "temperature": None,
            "wind_speed": None,
            "wind_direction": None,
            "humidity": None,
            "pressure": None,
            "forecast": [],
        }

    def test_missing_entity(self):
        """A missing entity returns the default result."""
        result = _make_reader().read_met_weather("weather.met")
        assert result["temperature"] is None
        assert result["forecast"] == []

    def test_full_attributes(self):
        """All Met.no attributes are surfaced."""
        forecast = [{"datetime": "x"}]
        attrs = {
            "wind_speed": 5.0,
            "wind_bearing": 180,
            "humidity": 50,
            "pressure": 1010,
            "forecast": forecast,
        }
        state = _state("15.5", attrs)

        result = _make_reader({"weather.met": state}).read_met_weather("weather.met")

        assert result["temperature"] == 15.5
        assert result["wind_speed"] == 5.0
        assert result["wind_direction"] == 180
        assert result["humidity"] == 50
        assert result["pressure"] == 1010
        assert result["forecast"] == forecast

    def test_no_forecast(self):
        """A missing forecast attribute yields an empty list."""
        state = _state("15.5", {})

        result = _make_reader({"weather.met": state}).read_met_weather("weather.met")

        assert result["forecast"] == []


class TestAsyncReadWeatherForecast:
    """Tests for the module-level async_read_weather_forecast."""

    @pytest.mark.asyncio
    async def test_missing_entity(self):
        """A missing entity returns None."""
        hass = Mock()
        hass.states.get.return_value = None

        result = await async_read_weather_forecast(hass, "weather.missing")

        assert result is None

    @pytest.mark.asyncio
    async def test_success(self):
        """A valid response returns the forecast list."""
        forecast_list = [{"datetime": "x"}]
        hass = Mock()
        hass.states.get.return_value = _state("20.0")
        hass.services.async_call = AsyncMock(
            return_value={"weather.test": {"forecast": forecast_list}}
        )

        result = await async_read_weather_forecast(hass, "weather.test")

        assert result == forecast_list

    @pytest.mark.asyncio
    async def test_empty_response(self):
        """A falsy service response returns None."""
        hass = Mock()
        hass.states.get.return_value = _state("20.0")
        hass.services.async_call = AsyncMock(return_value=None)

        result = await async_read_weather_forecast(hass, "weather.test")

        assert result is None

    @pytest.mark.asyncio
    async def test_entity_not_in_response(self):
        """A response missing the entity key returns None."""
        hass = Mock()
        hass.states.get.return_value = _state("20.0")
        hass.services.async_call = AsyncMock(return_value={"other": {}})

        result = await async_read_weather_forecast(hass, "weather.test")

        assert result is None

    @pytest.mark.asyncio
    async def test_non_dict_forecast_data(self):
        """A non-dict forecast data payload returns None."""
        hass = Mock()
        hass.states.get.return_value = _state("20.0")
        hass.services.async_call = AsyncMock(
            return_value={"weather.test": "not-a-dict"}
        )

        result = await async_read_weather_forecast(hass, "weather.test")

        assert result is None

    @pytest.mark.asyncio
    async def test_empty_forecast_list(self):
        """An empty forecast list returns None."""
        hass = Mock()
        hass.states.get.return_value = _state("20.0")
        hass.services.async_call = AsyncMock(
            return_value={"weather.test": {"forecast": []}}
        )

        result = await async_read_weather_forecast(hass, "weather.test")

        assert result is None

    @pytest.mark.asyncio
    async def test_forecast_not_a_list(self):
        """A non-list forecast returns None."""
        hass = Mock()
        hass.states.get.return_value = _state("20.0")
        hass.services.async_call = AsyncMock(
            return_value={"weather.test": {"forecast": "nope"}}
        )

        result = await async_read_weather_forecast(hass, "weather.test")

        assert result is None

    @pytest.mark.asyncio
    async def test_service_exception(self):
        """A service exception is caught and returns None."""
        hass = Mock()
        hass.states.get.return_value = _state("20.0")
        hass.services.async_call = AsyncMock(side_effect=RuntimeError("boom"))

        result = await async_read_weather_forecast(hass, "weather.test")

        assert result is None
