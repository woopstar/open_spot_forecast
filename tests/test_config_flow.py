"""Tests for the Open Spot Forecast config and options flows."""

from typing import Any
from unittest.mock import MagicMock, Mock

import pytest

from custom_components.open_spot_forecast.config_flow import (
    OpenSpotForecastConfigFlow,
    OpenSpotForecastOptionsFlow,
)
from custom_components.open_spot_forecast.const import CONF_REGION, DEFAULT_REGION


def _config_flow() -> Any:
    """Build a config flow with its form/entry helpers stubbed out."""
    flow: Any = OpenSpotForecastConfigFlow()
    flow.hass = Mock()
    flow.async_show_form = Mock(return_value={"type": "show_form"})
    flow.async_create_entry = Mock(return_value={"type": "create_entry"})
    return flow


def _options_flow() -> Any:
    """Build an options flow with a config entry and helpers stubbed out."""
    entry = MagicMock()
    entry.data = {CONF_REGION: "DK1"}
    entry.options = {}

    hass = Mock()
    hass.config_entries.async_get_known_entry.return_value = entry

    flow: Any = OpenSpotForecastOptionsFlow()
    flow.hass = hass
    flow.handler = "test_entry"
    flow.async_show_form = Mock(return_value={"type": "show_form"})
    flow.async_create_entry = Mock(return_value={"type": "create_entry"})
    return flow


# --- async_step_user ---------------------------------------------------------


@pytest.mark.asyncio
async def test_step_user_no_input_shows_form():
    """With no input the initial form is shown."""
    flow = _config_flow()
    result = await flow.async_step_user()

    assert result == {"type": "show_form"}
    flow.async_show_form.assert_called_once()
    kwargs = flow.async_show_form.call_args.kwargs
    assert kwargs["step_id"] == "user"
    assert kwargs["errors"] == {}
    assert kwargs["data_schema"] is not None


@pytest.mark.asyncio
async def test_step_user_invalid_region_sets_error():
    """An unknown region sets the invalid_region error and re-shows the form."""
    flow = _config_flow()
    await flow.async_step_user({CONF_REGION: "XX"})

    assert flow._errors == {"base": "invalid_region"}
    flow.async_show_form.assert_called_once()
    assert flow.async_show_form.call_args.kwargs["errors"] == {"base": "invalid_region"}


@pytest.mark.asyncio
async def test_step_user_valid_region_moves_to_sensors():
    """A valid region stores the input and advances to the sensor step."""
    flow = _config_flow()
    result = await flow.async_step_user({CONF_REGION: "DK1"})

    assert flow._data == {CONF_REGION: "DK1"}
    flow.async_show_form.assert_called_once()
    assert flow.async_show_form.call_args.kwargs["step_id"] == "sensors"
    assert result == {"type": "show_form"}


# --- async_step_sensors ------------------------------------------------------


@pytest.mark.asyncio
async def test_step_sensors_no_input_shows_form():
    """With no input the sensor form is shown."""
    flow = _config_flow()
    flow._data = {CONF_REGION: "DK1"}
    result = await flow.async_step_sensors()

    assert result == {"type": "show_form"}
    flow.async_show_form.assert_called_once()
    kwargs = flow.async_show_form.call_args.kwargs
    assert kwargs["step_id"] == "sensors"
    assert kwargs["errors"] == {}
    assert kwargs["data_schema"] is not None


@pytest.mark.asyncio
async def test_step_sensors_input_creates_entry():
    """Sensor input is merged and used to create the config entry."""
    flow = _config_flow()
    flow._data = {CONF_REGION: "DK1"}
    result = await flow.async_step_sensors({"enable_ml_prediction": True})

    assert result == {"type": "create_entry"}
    flow.async_create_entry.assert_called_once()
    kwargs = flow.async_create_entry.call_args.kwargs
    assert kwargs["title"] == "Open Spot Forecast DK1"
    assert kwargs["data"] == {CONF_REGION: "DK1", "enable_ml_prediction": True}


# --- async_get_options_flow --------------------------------------------------


def test_get_options_flow_returns_options_flow():
    """The config flow returns an options flow instance."""
    options_flow = OpenSpotForecastConfigFlow.async_get_options_flow(Mock())
    assert isinstance(options_flow, OpenSpotForecastOptionsFlow)


# --- async_step_init ---------------------------------------------------------


@pytest.mark.asyncio
async def test_options_init_no_input_shows_form():
    """With no input the options form is shown."""
    flow = _options_flow()
    result = await flow.async_step_init()

    assert result == {"type": "show_form"}
    flow.async_show_form.assert_called_once()
    kwargs = flow.async_show_form.call_args.kwargs
    assert kwargs["step_id"] == "init"
    assert kwargs["errors"] == {}
    assert kwargs["data_schema"] is not None


@pytest.mark.asyncio
async def test_options_init_input_creates_entry():
    """Options input is saved with a title from the config entry region."""
    flow = _options_flow()
    result = await flow.async_step_init({"vat": 0.25})

    assert result == {"type": "create_entry"}
    flow.async_create_entry.assert_called_once()
    kwargs = flow.async_create_entry.call_args.kwargs
    assert kwargs["title"] == "DK1"
    assert kwargs["data"] == {"vat": 0.25}


@pytest.mark.asyncio
async def test_options_init_falls_back_to_default_region():
    """Without a region in the entry data the default region is used."""
    flow = _options_flow()
    flow.config_entry.data = {}
    await flow.async_step_init({"vat": 0.25})

    assert flow.async_create_entry.call_args.kwargs["title"] == DEFAULT_REGION
