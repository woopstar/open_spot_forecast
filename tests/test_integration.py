"""Tests for Open Spot Forecast integration."""

from datetime import datetime
from pathlib import Path
from tempfile import mkdtemp
from unittest.mock import Mock

import pytest

from custom_components.open_spot_forecast.ml.predictor import SpotPricePredictor


def _make_hass() -> Mock:
    """Return a mock Home Assistant with a real storage path."""
    hass = Mock()
    hass.config.path.return_value = str(Path(mkdtemp()) / ".storage")
    return hass


class TestSpotPricePredictor:
    """Test the ML predictor."""

    def test_init(self):
        """Test predictor initialization."""
        hass = _make_hass()
        predictor = SpotPricePredictor(hass, "DK1")

        assert predictor.region == "DK1"
        assert predictor.is_trained is False
        assert predictor.predictions == []
        assert predictor.confidence_scores == []

    def test_wind_power_curve(self):
        """Test wind turbine power curve."""
        hass = _make_hass()
        predictor = SpotPricePredictor(hass, "DK1")

        # Below cut-in speed
        assert predictor._wind_power_curve(2) == 0

        # Between cut-in and rated
        power = predictor._wind_power_curve(8)
        assert 0 < power < 1

        # At rated speed
        assert predictor._wind_power_curve(12) == 1.0

        # Above rated but below cut-out
        assert predictor._wind_power_curve(20) == 1.0

        # Above cut-out
        assert predictor._wind_power_curve(30) == 0

    def test_extract_wind_features(self):
        """Test wind feature extraction."""
        hass = _make_hass()
        predictor = SpotPricePredictor(hass, "DK1")

        # Empty weather data
        features = predictor._extract_wind_features({})
        assert features["wind_speed_mean"] == 0
        assert features["wind_power_estimate"] == 0

        # With wind forecast
        weather_data = {
            "wind_forecast": [
                {"wind_speed": 5},
                {"wind_speed": 10},
                {"wind_speed": 15},
            ]
        }
        features = predictor._extract_wind_features(weather_data)

        assert features["wind_speed_mean"] == 10
        assert features["wind_speed_max"] == 15
        assert features["wind_power_estimate"] > 0

    def test_extract_solar_features(self):
        """Test solar feature extraction."""
        hass = _make_hass()
        predictor = SpotPricePredictor(hass, "DK1")

        # Empty weather data
        features = predictor._extract_solar_features({})
        assert features["solar_radiation_mean"] == 0
        assert features["solar_power_estimate"] == 0

        # With solar forecast
        weather_data = {
            "solar_forecast": [
                {"radiation": 500, "cloud_cover": 0.2},
                {"radiation": 600, "cloud_cover": 0.3},
                {"radiation": 700, "cloud_cover": 0.1},
            ]
        }
        features = predictor._extract_solar_features(weather_data)

        assert features["solar_radiation_mean"] == 600
        assert features["solar_power_estimate"] > 0

    def test_generate_time_features(self):
        """Test time feature generation."""
        hass = _make_hass()
        predictor = SpotPricePredictor(hass, "DK1")

        # 1 day, 60-minute intervals
        features = predictor._generate_time_features(1, 60)
        assert len(features) == 24

        # 2 days, 15-minute intervals
        features = predictor._generate_time_features(2, 15)
        assert len(features) == 2 * 96  # 2 days × 96 intervals

        # Check feature structure
        assert "hour" in features[0]
        assert "day_of_week" in features[0]
        assert "is_weekend" in features[0]
        assert "hour_sin" in features[0]
        assert "hour_cos" in features[0]

    def test_extract_hourly_pattern(self):
        """Test hourly pattern extraction."""
        hass = _make_hass()
        predictor = SpotPricePredictor(hass, "DK1")

        # Not enough data
        pattern = predictor._extract_hourly_pattern([100, 200])
        assert len(pattern) == 24
        assert all(p == 1.0 for p in pattern)

        # With 48 hours of data (2 days)
        prices = list(range(48))
        pattern = predictor._extract_hourly_pattern(prices)
        assert len(pattern) == 24
        assert all(isinstance(p, float) for p in pattern)

    def test_estimate_confidence(self):
        """Test confidence estimation."""
        hass = _make_hass()
        predictor = SpotPricePredictor(hass, "DK1")

        # Good features
        feature = {
            "wind_speed_mean": 10,
            "solar_radiation_mean": 500,
            "is_weekend": 0,
            "timestamp": datetime.now().isoformat(),
        }
        confidence = predictor._estimate_confidence(feature)
        assert 0.7 <= confidence <= 1.0

        # Missing weather data
        feature = {
            "wind_speed_mean": 0,
            "solar_radiation_mean": 0,
            "is_weekend": 0,
            "timestamp": datetime.now().isoformat(),
        }
        confidence = predictor._estimate_confidence(feature)
        assert confidence < 0.8

        # Weekend
        feature = {
            "wind_speed_mean": 10,
            "solar_radiation_mean": 500,
            "is_weekend": 1,
            "timestamp": datetime.now().isoformat(),
        }
        confidence = predictor._estimate_confidence(feature)
        assert confidence < 0.8

    def test_heuristic_predictions(self):
        """Test heuristic prediction generation."""
        hass = _make_hass()
        predictor = SpotPricePredictor(hass, "DK1")

        historical_prices = [100 + i for i in range(48)]  # 48 hours

        predictor._generate_heuristic_predictions(historical_prices, 2, 60)

        # Should generate 2 days × 24 hours = 48 predictions
        assert len(predictor.predictions) == 48
        assert len(predictor.confidence_scores) == 48

        # Check prediction structure
        for pred in predictor.predictions:
            assert "start" in pred
            assert "price" in pred
            assert "confidence" in pred
            assert pred["price"] >= 0
            assert 0.3 <= pred["confidence"] <= 1.0

    def test_predict_with_insufficient_data(self):
        """Test prediction with insufficient training data."""
        hass = _make_hass()
        predictor = SpotPricePredictor(hass, "DK1")

        weather_data = {}
        historical_prices = [100, 200, 300]  # Less than 24 hours

        predictor.predict(weather_data, historical_prices, 1, 60)

        # Should use heuristic fallback
        assert len(predictor.predictions) > 0
        assert predictor.is_trained is False

    def test_get_predictions_for_day(self):
        """Test getting predictions for a specific day."""
        hass = _make_hass()
        predictor = SpotPricePredictor(hass, "DK1")

        # Generate some predictions
        historical_prices = [100 + i for i in range(48)]
        predictor._generate_heuristic_predictions(historical_prices, 3, 60)

        # Get predictions for today
        today_preds = predictor.get_predictions_for_day(0)
        assert len(today_preds) > 0

        # Get predictions for tomorrow
        tomorrow_preds = predictor.get_predictions_for_day(1)
        assert len(tomorrow_preds) > 0

    def test_get_prediction_stats(self):
        """Test prediction statistics."""
        hass = _make_hass()
        predictor = SpotPricePredictor(hass, "DK1")

        # No predictions
        stats = predictor.get_prediction_stats()
        assert stats == {}

        # With predictions
        historical_prices = [100 + i for i in range(48)]
        predictor._generate_heuristic_predictions(historical_prices, 2, 60)

        stats = predictor.get_prediction_stats()
        assert "min_price" in stats
        assert "max_price" in stats
        assert "mean_price" in stats
        assert "mean_confidence" in stats
        assert stats["total_predictions"] == 48


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
