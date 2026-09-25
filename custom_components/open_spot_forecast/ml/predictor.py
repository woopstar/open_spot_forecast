"""Machine learning predictor for spot prices."""

import logging
import threading
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np

from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

from ..price_series import is_invalid_price_series, known_prices
from .features import FeatureMixin
from .lead_time import LeadTimeMixin
from .learning import LearningMixin
from .models import ModelMixin, create_price_model
from .numpy_models import NumpyRandomForest
from .retraining import RetrainMixin
from .storage import LearningStorage

_LOGGER = logging.getLogger(__name__)


class SpotPricePredictor(
    FeatureMixin, ModelMixin, LearningMixin, LeadTimeMixin, RetrainMixin
):
    """ML-based spot price predictor using weather and historical price data."""

    def __init__(
        self, hass: HomeAssistant, region: str, tz_name: str = "Europe/Copenhagen"
    ):
        """Initialize the predictor."""
        self.hass = hass
        self.region = region
        self.tz = dt_util.get_time_zone(tz_name) or UTC
        self.predictions = []
        self.confidence_scores = []

        # ML models (pure numpy implementations)
        self.wind_model = NumpyRandomForest(
            n_estimators=100, max_depth=10, random_state=42
        )
        self.solar_model = NumpyRandomForest(
            n_estimators=100, max_depth=10, random_state=42
        )
        self.price_model = create_price_model()

        # Feature scalers (simple numpy-based scaling)
        self.wind_scaler = {"mean": 0, "std": 1}
        self.solar_scaler = {"mean": 0, "std": 1}
        self.price_scaler = {"mean": 0, "std": 1}

        # Solar scaling: learned ratio of actual_power / solcast_estimate
        self.solar_scale = 1.0
        self._solar_scale_samples = 0

        # Training status
        self.is_trained = False
        self.training_samples = 0

        # Retrain tracking (see retraining.py): retrain when training inputs
        # changed after last_trained_at; HPO counts new days of price data
        self.last_trained_at: datetime | None = None
        self._prices_updated_at: datetime | None = None
        self._hpo_counter = 0

        # A retrain can take minutes and forecast runs can overlap, so
        # serialize predict() — two threads must never fit the same model
        self._predict_lock = threading.Lock()

        # Self-learning: error tracking (predictions stored in SQLite)
        self.error_metrics = {}  # Track prediction errors by hour
        self.bias_correction = {}  # Hourly bias correction factors
        self.volatility_mae: dict[int, float] = {}  # Per-slot volatility (EMA of MAE)
        self.learning_rate = 0.1  # Adaptive learning rate
        self.price_history = []  # Store historical prices for multi-day training
        self.max_history_days = 30  # Keep 30 days of history
        # Live MAE/RMSE per lead-time bucket (see lead_time.py)
        self.lead_time_accuracy: dict[str, dict[str, float | int]] = {}

        # Storage for persistence
        self.storage = LearningStorage(hass, region)

        # Note: Learning data is loaded asynchronously in __init__.py

    def predict(
        self,
        weather_data: dict,
        historical_prices: Sequence[float | None],
        forecast_days: int = 7,
        interval_minutes: int = 15,
        known_data_end_time: datetime | None = None,
    ) -> None:
        """Generate price predictions using ML models.

        Args:
            weather_data: Dictionary containing wind and solar forecasts
            historical_prices: List of historical spot prices
            forecast_days: Number of days to forecast (1-7)
            interval_minutes: Prediction interval in minutes (default 15)
            known_data_end_time: Timestamp after which to start predicting.
                No predictions will be generated for times we already
                have actual/committed prices for. If None, starts at
                the current slot (see first_prediction_slot).
        """
        with self._predict_lock:
            self._predict(
                weather_data,
                historical_prices,
                forecast_days,
                interval_minutes,
                known_data_end_time,
            )

    def _predict(
        self,
        weather_data: dict,
        historical_prices: Sequence[float | None],
        forecast_days: int,
        interval_minutes: int,
        known_data_end_time: datetime | None,
    ) -> None:
        """Generate price predictions; the caller holds _predict_lock."""
        try:
            _LOGGER.info(
                "Starting ML prediction with %d historical prices, forecast_days=%d",
                len(historical_prices),
                forecast_days,
            )

            # Validate input data (None marks a slot missing in the source)
            if not known_prices(historical_prices):
                _LOGGER.warning(
                    "No historical prices provided, cannot generate predictions"
                )
                self.predictions = []
                self.confidence_scores = []
                return

            # All-zero or non-finite prices come from a failing source: don't
            # store, train or predict on them, keep the previous predictions
            if is_invalid_price_series(historical_prices):
                _LOGGER.warning(
                    "Known prices are all zero or not finite, "
                    "keeping the previous predictions"
                )
                return

            # Extract features from weather data
            wind_features = self._extract_wind_features(weather_data)
            solar_features = self._extract_solar_features(weather_data)

            # Learn solar scaling factor: how does actual output compare to Solcast?
            actual_solar = weather_data.get("solar_power")
            solcast_estimate = solar_features.get("solar_power_estimate", 0)
            if (
                actual_solar
                and solcast_estimate
                and actual_solar > 0
                and solcast_estimate > 0
            ):
                ratio = float(actual_solar) / float(solcast_estimate)
                if 0.1 < ratio < 10.0:  # Sanity check
                    self._solar_scale_samples += 1
                    alpha = min(0.3, 1.0 / max(1, self._solar_scale_samples))
                    self.solar_scale = (1 - alpha) * self.solar_scale + alpha * ratio
                    _LOGGER.debug(
                        "Solar scale updated: %.3f (actual=%.0f, estimate=%.0f, samples=%d)",
                        self.solar_scale,
                        actual_solar,
                        solcast_estimate,
                        self._solar_scale_samples,
                    )
                # Apply scaling to solar features
                solar_features["solar_power_estimate"] = (
                    solar_features["solar_power_estimate"] * self.solar_scale
                )
                solar_features["solar_radiation_mean"] = (
                    solar_features.get("solar_radiation_mean", 0) * self.solar_scale
                )

            # Generate time-based features
            time_features = self._generate_time_features(
                forecast_days, interval_minutes, known_data_end_time
            )

            # Combine all features
            all_features = self._combine_features(
                wind_features,
                solar_features,
                time_features,
                historical_prices,
                weather_data,
            )

            _LOGGER.info(
                "Generated %d feature sets, model trained: %s",
                len(all_features),
                self.is_trained,
            )

            # Retrain only if the model is missing or its inputs changed
            if len(known_prices(historical_prices)) > 24:
                self.record_training_prices(historical_prices)
                if self.needs_retraining():
                    _LOGGER.info(
                        "Training ML models (data updated %s, last trained %s)",
                        self.last_data_update,
                        self.last_trained_at,
                    )
                    self.retrain(historical_prices, all_features)
                else:
                    _LOGGER.debug("No new training data since %s", self.last_trained_at)

            # Generate predictions
            if self.is_trained:
                _LOGGER.info("Using trained ML model for predictions")
                self._generate_predictions(
                    all_features, forecast_days, interval_minutes
                )
            else:
                # Fallback to simple heuristic if not enough training data
                _LOGGER.info("Using heuristic fallback for predictions")
                self._generate_heuristic_predictions(
                    historical_prices,
                    forecast_days,
                    interval_minutes,
                    known_data_end_time,
                )

            _LOGGER.info(
                "Generated %d predictions for %d days",
                len(self.predictions),
                forecast_days,
            )

            # Log sample predictions
            if self.predictions:
                sample = self.predictions[0]
                _LOGGER.info(
                    "Sample prediction: start=%s, price=%.4f, confidence=%.2f",
                    sample.get("start"),
                    sample.get("price", 0),
                    sample.get("confidence", 0),
                )

        except Exception as err:
            _LOGGER.error("Error generating ML predictions: %s", err, exc_info=True)
            self.predictions = []
            self.confidence_scores = []

    def get_predictions_for_day(self, day_offset: int = 0) -> list[dict]:
        """Get predictions for a specific day."""
        target_date = dt_util.now() + timedelta(days=day_offset)
        target_str = target_date.strftime("%Y-%m-%d")

        return [
            p for p in self.predictions if p.get("start", "").startswith(target_str)
        ]

    def get_prediction_stats(self) -> dict[str, Any]:
        """Get statistics for predictions."""
        if not self.predictions:
            return {}

        prices = [p["price"] for p in self.predictions]
        confidences = [p["confidence"] for p in self.predictions]

        return {
            "min_price": min(prices),
            "max_price": max(prices),
            "mean_price": float(np.mean(prices)),
            "mean_confidence": float(np.mean(confidences)),
            "total_predictions": len(prices),
            "is_ml_model": self.is_trained,
            "training_samples": self.training_samples,
        }
