"""Tests for sensor attribute size and caching behaviour."""

from unittest.mock import MagicMock

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
        MagicMock(), _entry(), api_data, "DK1", "DKK", 0.25, 2, "kWh"
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


def _ml_sensor(api_data: dict) -> MLPredictionSensor:
    return MLPredictionSensor(MagicMock(), _entry(), api_data, "DKK", 0.25, 2, "kWh")


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
