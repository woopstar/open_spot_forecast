"""Tests for the Open Spot Forecast binary sensor platform."""

from unittest.mock import MagicMock, Mock

import pytest

from custom_components.open_spot_forecast.binary_sensor import (
    MLModelTrainedSensor,
    TomorrowAvailableSensor,
    async_setup_entry,
)
from custom_components.open_spot_forecast.const import DOMAIN


def _entry(entry_id: str = "test") -> MagicMock:
    """Build a mock config entry with the given entry id."""
    entry = MagicMock()
    entry.entry_id = entry_id
    return entry


def _hass() -> Mock:
    """Build a mock Home Assistant with a real dispatcher data dict."""
    hass = Mock()
    hass.data = {}
    return hass


# --- async_setup_entry -------------------------------------------------------


@pytest.mark.asyncio
async def test_async_setup_entry_adds_two_sensors():
    """Setup registers a tomorrow-availability and an ML-trained sensor."""
    api_data = {"tomorrow_available": True}
    hass = Mock()
    hass.data = {DOMAIN: {"test": api_data}}
    entry = _entry()
    async_add_entities = Mock()

    await async_setup_entry(hass, entry, async_add_entities)

    async_add_entities.assert_called_once()
    sensors, update_before_add = async_add_entities.call_args.args
    assert len(sensors) == 2
    assert isinstance(sensors[0], TomorrowAvailableSensor)
    assert isinstance(sensors[1], MLModelTrainedSensor)
    assert update_before_add is True


# --- TomorrowAvailableSensor -------------------------------------------------


def test_tomorrow_available_sensor_is_on():
    """The sensor reports on when all of tomorrow's 96 slots are known."""
    sensor = TomorrowAvailableSensor(_hass(), _entry(), {"prices_tomorrow": [1.0] * 96})
    assert sensor.is_on is True


def test_tomorrow_available_sensor_is_off():
    """Missing or partial prices (the old >= 23 threshold) are not available."""
    partial = TomorrowAvailableSensor(
        _hass(), _entry(), {"prices_tomorrow": [1.0] * 23}
    )
    assert partial.is_on is False

    sensor_missing = TomorrowAvailableSensor(_hass(), _entry(), {})
    assert sensor_missing.is_on is False


def test_tomorrow_available_sensor_attributes():
    """Attributes show tomorrow's price count against a full day's slots."""
    sensor = TomorrowAvailableSensor(_hass(), _entry(), {"prices_tomorrow": [1.0] * 40})
    assert sensor.extra_state_attributes == {
        "tomorrow_prices_count": 40,
        "tomorrow_slots_expected": 96,
    }

    empty = TomorrowAvailableSensor(_hass(), _entry(), {})
    assert empty.extra_state_attributes["tomorrow_prices_count"] == 0


def test_tomorrow_available_sensor_identity():
    """Unique id, name and device info are derived from the entry."""
    entry = _entry("abc")
    sensor = TomorrowAvailableSensor(_hass(), entry, {})
    assert sensor._attr_unique_id == "open_spot_forecast_abc_tomorrow_available"
    assert sensor._attr_name == "Tomorrow Prices Available"
    assert sensor._attr_device_info == {"identifiers": {(DOMAIN, "abc")}}


@pytest.mark.asyncio
async def test_tomorrow_available_sensor_added_to_hass():
    """The sensor registers a dispatcher callback when added to hass."""
    hass = _hass()
    sensor = TomorrowAvailableSensor(hass, _entry(), {})
    await sensor.async_added_to_hass()
    # The dispatcher connection stores its callback table in hass.data.
    assert len(hass.data) == 1


@pytest.mark.asyncio
async def test_tomorrow_available_sensor_handle_update():
    """An update request triggers a state write."""
    sensor = TomorrowAvailableSensor(_hass(), _entry(), {})
    sensor.async_write_ha_state = Mock()
    await sensor._handle_update()
    sensor.async_write_ha_state.assert_called_once()


# --- MLModelTrainedSensor ----------------------------------------------------


def test_ml_model_trained_sensor_is_on():
    """The sensor reports on when the ML predictor is trained."""
    predictor = MagicMock()
    predictor.is_trained = True
    sensor = MLModelTrainedSensor(_hass(), _entry(), {"ml_predictor": predictor})
    assert sensor.is_on is True


def test_ml_model_trained_sensor_is_off():
    """The sensor reports off when the predictor is untrained or missing."""
    untrained = MagicMock()
    untrained.is_trained = False
    sensor = MLModelTrainedSensor(_hass(), _entry(), {"ml_predictor": untrained})
    assert sensor.is_on is False

    missing = MLModelTrainedSensor(_hass(), _entry(), {})
    assert missing.is_on is False


def test_ml_model_trained_sensor_attributes_empty():
    """Without a predictor no extra attributes are exposed."""
    sensor = MLModelTrainedSensor(_hass(), _entry(), {})
    assert sensor.extra_state_attributes == {}


def test_ml_model_trained_sensor_attributes_with_predictor():
    """A predictor exposes training samples and total predictions."""
    predictor = MagicMock()
    predictor.training_samples = 42
    predictor.predictions = [1, 2, 3]
    sensor = MLModelTrainedSensor(_hass(), _entry(), {"ml_predictor": predictor})
    assert sensor.extra_state_attributes == {
        "training_samples": 42,
        "total_predictions": 3,
    }


def test_ml_model_trained_sensor_identity():
    """Unique id, name and device info are derived from the entry."""
    entry = _entry("abc")
    sensor = MLModelTrainedSensor(_hass(), entry, {})
    assert sensor._attr_unique_id == "open_spot_forecast_abc_ml_trained"
    assert sensor._attr_name == "ML Model Trained"
    assert sensor._attr_device_info == {"identifiers": {(DOMAIN, "abc")}}


@pytest.mark.asyncio
async def test_ml_model_trained_sensor_added_to_hass():
    """The sensor registers a dispatcher callback when added to hass."""
    hass = _hass()
    sensor = MLModelTrainedSensor(hass, _entry(), {})
    await sensor.async_added_to_hass()
    assert len(hass.data) == 1


@pytest.mark.asyncio
async def test_ml_model_trained_sensor_handle_update():
    """An update request triggers a state write."""
    sensor = MLModelTrainedSensor(_hass(), _entry(), {})
    sensor.async_write_ha_state = Mock()
    await sensor._handle_update()
    sensor.async_write_ha_state.assert_called_once()
