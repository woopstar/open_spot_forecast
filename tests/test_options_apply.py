"""Saved options are what the integration uses (#245, #239).

The options form suggested the developer's own weather and solar entities
and saved them back when a field was left or cleared. The integration read
its sensors at setup only and had no update listener, so saved options did
not reach a running entry either.
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from custom_components.open_spot_forecast import async_reload_entry
from custom_components.open_spot_forecast.config_flow import (
    OPTIONAL_SENSOR_KEYS,
    OpenSpotForecastConfigFlow,
    OpenSpotForecastOptionsFlow,
)
from custom_components.open_spot_forecast.const import (
    CONF_REGION,
    CONF_SOLAR_FORECAST_SENSOR,
    CONF_TEMPERATURE_SENSOR,
    CONF_WIND_SPEED_SENSOR,
)
from custom_components.open_spot_forecast.sensor_entities import SensorEntities

DEVELOPER_ENTITIES = (
    "weather.forecast_mellemlokken_23",
    "sensor.power_inverter_input_total",
    "sensor.solcast_pv_forecast_forecast_today",
    "sensor.metroair_330_outdoor_temperature",
)


def _entry(data: dict[str, Any], options: dict[str, Any]) -> MagicMock:
    entry = MagicMock()
    entry.data = {CONF_REGION: "SE3", **data}
    entry.options = options
    return entry


def _options_flow(entry: MagicMock) -> Any:
    hass = Mock()
    hass.config_entries.async_get_known_entry.return_value = entry
    flow: Any = OpenSpotForecastOptionsFlow()
    flow.hass = hass
    flow.handler = "test_entry"
    flow.async_show_form = Mock(return_value={"type": "show_form"})
    flow.async_create_entry = Mock(return_value={"type": "create_entry"})
    return flow


def _form(flow: Any) -> dict[str, tuple[Any, Any]]:
    """Return each sensor field's default and suggested value in the shown form."""
    schema = flow.async_show_form.call_args.kwargs["data_schema"]
    fields: dict[str, tuple[Any, Any]] = {}
    for key in schema.schema:
        if str(key) in OPTIONAL_SENSOR_KEYS:
            default = key.default() if callable(key.default) else None
            suggested = (key.description or {}).get("suggested_value")
            fields[str(key)] = (default, suggested)
    return fields


@pytest.mark.asyncio
async def test_the_options_form_suggests_the_saved_sensors_only() -> None:
    entry = _entry(
        {CONF_TEMPERATURE_SENSOR: "sensor.setup_temperature"},
        {CONF_WIND_SPEED_SENSOR: "weather.openweathermap"},
    )
    flow = _options_flow(entry)

    await flow.async_step_init()
    fields = _form(flow)

    assert fields[CONF_WIND_SPEED_SENSOR] == (None, "weather.openweathermap")
    assert fields[CONF_TEMPERATURE_SENSOR] == (None, "sensor.setup_temperature")
    assert fields[CONF_SOLAR_FORECAST_SENSOR] == (None, None)
    assert not set(DEVELOPER_ENTITIES) & {v for pair in fields.values() for v in pair}


@pytest.mark.asyncio
async def test_the_setup_form_suggests_no_entity() -> None:
    flow: Any = OpenSpotForecastConfigFlow()
    flow.hass = Mock()
    flow.async_show_form = Mock(return_value={"type": "show_form"})
    flow._data = {CONF_REGION: "SE3"}

    await flow.async_step_sensors()

    assert set(_form(flow).values()) == {(None, None)}


@pytest.mark.asyncio
async def test_a_cleared_sensor_stays_cleared() -> None:
    """An emptied picker is left out of the input; None is saved (#239)."""
    entry = _entry(
        {CONF_SOLAR_FORECAST_SENSOR: "sensor.solcast_pv_forecast_forecast_today"},
        {CONF_TEMPERATURE_SENSOR: "sensor.ute_temperatur_67an"},
    )
    flow = _options_flow(entry)

    await flow.async_step_init({CONF_WIND_SPEED_SENSOR: "weather.openweathermap"})
    saved = flow.async_create_entry.call_args.kwargs["data"]

    assert saved[CONF_SOLAR_FORECAST_SENSOR] is None
    assert saved[CONF_TEMPERATURE_SENSOR] is None
    entry.options = saved
    sensors = SensorEntities.from_entry(entry)
    assert sensors.wind_speed == "weather.openweathermap"
    assert sensors.solar_forecast is None
    assert sensors.temperature is None


@pytest.mark.asyncio
async def test_saved_options_reload_the_entry(setup_entry: Any) -> None:
    ml_predictor = Mock(cross_border=None, predictions=[])
    ml_predictor._load_learning_data = AsyncMock()
    ml_predictor.save_learning_data = AsyncMock()
    prices = {"today": [1.0] * 96, "tomorrow": [], "raw_today": [], "raw_tomorrow": []}
    reader = Mock()
    reader.read_stromligning_sensor.return_value = prices
    reader.read_stromligning_tomorrow_sensor.return_value = {
        "tomorrow": [],
        "raw_tomorrow": [],
        "available": False,
    }
    reader.read_spot_prices.return_value = prices
    reader.read_weather_sensors.return_value = {}
    _, callbacks = await setup_entry(reader, ml_predictor)
    hass = Mock()
    hass.config_entries.async_reload = AsyncMock()
    entry = MagicMock(entry_id="test")

    await callbacks["async_reload_entry"](hass, entry)

    assert callbacks["async_reload_entry"] is async_reload_entry
    hass.config_entries.async_reload.assert_awaited_once_with("test")
