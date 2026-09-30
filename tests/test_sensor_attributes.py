"""Tests for sensor attribute size and caching behaviour."""

import json
import logging
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from homeassistant.components.recorder.db_schema import StateAttributes
from homeassistant.const import EVENT_STATE_CHANGED
from homeassistant.core import Event, EventStateChangedData, State

from custom_components.open_spot_forecast.const import (
    DETAILED_MAX_PREDICTION_HOURS,
    RECORDER_MAX_ATTRIBUTES_BYTES,
)
from custom_components.open_spot_forecast.forecast_attributes import attributes_size
from custom_components.open_spot_forecast.price_output import PriceOutput
from custom_components.open_spot_forecast.sensor import (
    LearningMetricsSensor,
    MLPredictionSensor,
    SpotPriceSensor,
)


def _entry() -> MagicMock:
    entry = MagicMock()
    entry.entry_id = "test_entry"
    return entry


def _spot_price_sensor(api_data: dict) -> SpotPriceSensor:
    return SpotPriceSensor(
        MagicMock(), _entry(), api_data, "DK1", "DKK", PriceOutput(precision=2)
    )


def test_spot_price_attributes_omit_raw_arrays():
    """Raw dict arrays are excluded to stay under HA's 16 KB attribute limit."""
    api_data = {
        "stromligning_data": {
            "today": [1.0, 2.0, 3.0],
            "tomorrow": [4.0, 5.0],
            "prices_15min": [{"price": 1.0, "timestamp": "2026-09-22T00:00:00Z"}],
            "raw_today": [{"price": 1.0, "timestamp": "2026-09-22T00:00:00Z"}],
            "raw_tomorrow": [{"price": 4.0, "timestamp": "2026-09-23T00:00:00Z"}],
        }
    }
    sensor = _spot_price_sensor(api_data)

    attrs = sensor.extra_state_attributes

    assert attrs["today_prices"] == [1.0, 2.0, 3.0]
    assert attrs["tomorrow_prices"] == [4.0, 5.0]
    assert attrs["price_source"] == "stromligning"
    assert "prices_15min" not in attrs
    assert "raw_today" not in attrs
    assert "raw_tomorrow" not in attrs


def _ml_sensor(api_data: dict, prediction_hours: int = 48) -> MLPredictionSensor:
    return MLPredictionSensor(
        MagicMock(),
        _entry(),
        api_data,
        "DKK",
        PriceOutput(precision=2),
        prediction_hours,
    )


def _predictions(count: int) -> list[dict]:
    return [
        {
            "start": "2026-09-22T00:00:00+02:00",
            "end": "2026-09-22T00:15:00+02:00",
            "price": 100.0 + i,
            "confidence": 0.8,
        }
        for i in range(count)
    ]


def test_ml_prediction_attributes_truncate_predictions():
    """Only the next 48 hours of predictions are exposed as attributes."""
    predictor = MagicMock()
    predictor.predictions = _predictions(200)
    predictor.get_prediction_stats.return_value = {
        "min_price": 100.0,
        "max_price": 200.0,
        "mean_price": 150.0,
        "mean_confidence": 0.8,
        "total_predictions": 200,
        "is_ml_model": True,
        "training_samples": 10,
    }
    sensor = _ml_sensor({"ml_predictor": predictor})

    attrs = sensor.extra_state_attributes

    assert len(attrs["predictions"]) == 192
    assert attrs["total_predictions"] == 200


def test_ml_prediction_attributes_configurable_hours():
    """The prediction window follows the configured hours in 12-hour steps."""
    for hours, expected_slots in [(12, 48), (24, 96), (36, 144), (72, 288)]:
        predictor = MagicMock()
        predictor.predictions = _predictions(300)
        predictor.get_prediction_stats.return_value = {}

        sensor = _ml_sensor({"ml_predictor": predictor}, prediction_hours=hours)

        assert len(sensor.extra_state_attributes["predictions"]) == expected_slots


def test_learning_metrics_cached_across_properties():
    """get_learning_metrics is computed once per state write."""
    predictor = MagicMock()
    predictor.get_learning_metrics.return_value = {
        "status": "learning",
        "total_samples": 42,
        "mae": 1.5,
    }
    sensor = LearningMetricsSensor(MagicMock(), _entry(), {"ml_predictor": predictor})

    assert sensor.native_value == 42
    attrs = sensor.extra_state_attributes
    assert attrs["status"] == "learning"

    # Both properties share a single computation.
    assert predictor.get_learning_metrics.call_count == 1


# --- What the recorder stores (#103) ---------------------------------------------------

NOW = datetime(2026, 9, 24, 10, 5, tzinfo=UTC)


def _recorded(sensor: Any, attributes: dict[str, Any]) -> bytes:
    """Return the attributes as Home Assistant's recorder encodes them.

    The state carries the entity's unrecorded attributes the way
    ``Entity.async_internal_added_to_hass`` sets them.
    """
    cls = type(sensor)
    unrecorded = (
        cls._entity_component_unrecorded_attributes | cls._unrecorded_attributes
    )
    state = State(
        "sensor.open_spot_forecast_dk1_test",
        "1.0",
        attributes,
        state_info={"unrecorded_attributes": unrecorded},
    )
    event: Event[EventStateChangedData] = Event(
        EVENT_STATE_CHANGED,
        {"entity_id": state.entity_id, "old_state": None, "new_state": state},
    )
    return StateAttributes.shared_attrs_bytes_from_event(event, None)


def _full_forecast(slots: int) -> list[dict[str, Any]]:
    """Return 15-min predictions with full-precision prices (the worst case)."""
    return [
        {
            "start": (NOW + timedelta(minutes=15 * i)).isoformat(),
            "end": (NOW + timedelta(minutes=15 * i + 15)).isoformat(),
            "price": -1.3 + (i * 0.0371937) % 4.8,
            "confidence": 0.3 + (i % 67) / 100,
        }
        for i in range(slots)
    ]


def test_the_forecast_window_is_recorded_without_its_series(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """72 detailed hours stay in the live state; the rest is recorded."""
    predictor = MagicMock()
    predictor.predictions = _full_forecast(7 * 96)
    predictor.get_prediction_stats.return_value = {
        "mean_confidence": 0.62,
        "total_predictions": 7 * 96,
        "is_ml_model": True,
        "training_samples": 5760,
    }
    sensor = MLPredictionSensor(
        MagicMock(),
        _entry(),
        {"ml_predictor": predictor},
        "DKK",
        PriceOutput(),
        DETAILED_MAX_PREDICTION_HOURS,
    )
    with patch("homeassistant.util.dt.utcnow", return_value=NOW):
        attrs = {**sensor.extra_state_attributes, "attribution": sensor.attribution}

    # The window is not shortened to fit
    assert len(attrs["predictions"]) == DETAILED_MAX_PREDICTION_HOURS * 4
    assert attributes_size(attrs) > RECORDER_MAX_ATTRIBUTES_BYTES

    with caplog.at_level(logging.WARNING):
        stored = json.loads(_recorded(sensor, attrs))

    assert "exceed maximum size" not in caplog.text
    assert "predictions" not in stored
    assert stored["forecast_mean"] == pytest.approx(attrs["forecast_mean"])
    assert stored["known_until"] == attrs["known_until"]


def test_the_learning_metrics_are_recorded_without_the_slot_metrics(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """All 96 slots stay in the live state; the scalar metrics are recorded."""
    slots = {
        str(slot): {
            "hour": slot // 4,
            "minute": (slot % 4) * 15,
            "mae": 0.123456789012345,
            "bias": -0.0123456789012345,
            "samples": 1234,
            "bias_correction": 0.0123456789012345,
            "volatility": 0.0987654321098765,
        }
        for slot in range(96)
    }
    predictor = MagicMock()
    predictor.get_learning_metrics.return_value = {
        "status": "learning",
        "message": "Actively learning from 118464 comparisons",
        "total_samples": 118464,
        "mae": 0.2,
        "rmse": 0.3,
        "hourly_metrics": slots,
    }
    sensor = LearningMetricsSensor(MagicMock(), _entry(), {"ml_predictor": predictor})
    attrs = sensor.extra_state_attributes

    assert len(attrs["hourly_metrics"]) == 96

    with caplog.at_level(logging.WARNING):
        stored = json.loads(_recorded(sensor, attrs))

    assert "exceed maximum size" not in caplog.text
    assert "hourly_metrics" not in stored
    assert stored["mae"] == pytest.approx(0.2)
    assert stored["status"] == "learning"
