"""Tests for the lead-time forecast accuracy diagnostic sensors."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from homeassistant.components.sensor import SensorStateClass
from homeassistant.const import EntityCategory
from homeassistant.util import slugify as util_slugify

from custom_components.open_spot_forecast.accuracy_sensor import (
    LeadTimeAccuracySensor,
    build_lead_time_accuracy_sensors,
)
from custom_components.open_spot_forecast.const import (
    DOMAIN,
    LEAD_TIME_WINDOW_DAYS,
    UPDATE_SIGNAL,
)

ACCURACY = {
    "day_1": {"mae": 0.12, "rmse": 0.2, "bias": -0.03, "samples": 40},
    "day_2": {"mae": 0.18, "rmse": 0.25, "bias": 0.01, "samples": 36},
}


def _entry() -> MagicMock:
    """Return a mock config entry."""
    entry = MagicMock()
    entry.entry_id = "test_entry"
    return entry


def _sensor(
    bucket: str = "day_1", metric: str = "mae", api_data: dict | None = None
) -> LeadTimeAccuracySensor:
    """Return an accuracy sensor backed by a predictor with ACCURACY cached."""
    if api_data is None:
        api_data = {"ml_predictor": Mock(lead_time_accuracy=ACCURACY)}
    return LeadTimeAccuracySensor(Mock(), _entry(), api_data, "DKK", 3, bucket, metric)


def test_build_creates_mae_and_rmse_per_bucket():
    """One MAE and one RMSE sensor exist for each lead-time bucket."""
    sensors = build_lead_time_accuracy_sensors(Mock(), _entry(), {}, "DKK", 3)

    assert [s.translation_key for s in sensors] == [
        "forecast_mae_day_1",
        "forecast_rmse_day_1",
        "forecast_mae_day_2",
        "forecast_rmse_day_2",
        "forecast_mae_day_3",
        "forecast_rmse_day_3",
        "forecast_mae_day_4_plus",
        "forecast_rmse_day_4_plus",
    ]
    assert len({s.unique_id for s in sensors}) == len(sensors)


def test_entity_attributes():
    """Sensors are diagnostic measurements with stable IDs and device info."""
    sensor = _sensor("day_2", "rmse")

    assert sensor.unique_id == f"{DOMAIN}_test_entry_forecast_rmse_day_2"
    assert sensor.translation_key == "forecast_rmse_day_2"
    assert sensor.has_entity_name is True
    assert sensor.entity_category is EntityCategory.DIAGNOSTIC
    assert sensor.state_class is SensorStateClass.MEASUREMENT
    assert sensor.native_unit_of_measurement == "DKK/kWh"
    assert sensor.suggested_display_precision == 3
    assert sensor.device_info == {"identifiers": {(DOMAIN, "test_entry")}}


@pytest.mark.parametrize(
    ("bucket", "metric", "expected"),
    [("day_1", "mae", 0.12), ("day_1", "rmse", 0.2), ("day_2", "mae", 0.18)],
)
def test_native_value_reads_cached_metric(bucket, metric, expected):
    """The state is the cached MAE/RMSE of the sensor's bucket."""
    assert _sensor(bucket, metric).native_value == pytest.approx(expected)


def test_extra_state_attributes():
    """Attributes expose the sample count, bias and window length."""
    assert _sensor("day_1", "mae").extra_state_attributes == {
        "samples": 40,
        "bias": pytest.approx(-0.03),
        "window_days": LEAD_TIME_WINDOW_DAYS,
    }


@pytest.mark.parametrize(
    "api_data",
    [{}, {"ml_predictor": None}, {"ml_predictor": Mock(lead_time_accuracy={})}],
)
def test_unknown_until_predictions_are_matched(api_data):
    """No predictor or no matched predictions yet means an unknown state."""
    sensor = _sensor("day_3", "mae", api_data)

    assert sensor.native_value is None
    assert sensor.extra_state_attributes == {
        "samples": 0,
        "bias": None,
        "window_days": LEAD_TIME_WINDOW_DAYS,
    }


@pytest.mark.asyncio
async def test_async_added_to_hass_listens_for_updates():
    """The sensor writes state on the update signal and unsubscribes on removal."""
    sensor = _sensor()
    unsub = Mock()
    sensor.async_on_remove = Mock()

    with patch(
        "custom_components.open_spot_forecast.accuracy_sensor.async_dispatcher_connect",
        return_value=unsub,
    ) as connect:
        await sensor.async_added_to_hass()

    connect.assert_called_once_with(
        sensor.hass, util_slugify(UPDATE_SIGNAL), sensor.async_write_ha_state
    )
    sensor.async_on_remove.assert_called_once_with(unsub)
