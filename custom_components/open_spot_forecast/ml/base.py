"""Shared type declarations for the SpotPricePredictor mixins.

The predictor is composed of three cooperating mixins (FeatureMixin,
ModelMixin, LearningMixin) whose methods freely reference each other's
attributes and methods via ``self``. mypy cannot see across those class
boundaries unless the shared surface is declared once. This base class
holds those declarations only — no runtime state or behaviour — so the
mixins inherit a consistent, fully-typed ``self``.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, tzinfo
from typing import Any

from homeassistant.core import HomeAssistant

from .numpy_models import NumpyGradientBoosting, NumpyRandomForest
from .storage import LearningStorage


class PredictorBase:
    """Type-only base declaring the predictor's shared state and calls."""

    # --- Instance state, initialized in SpotPricePredictor.__init__ ---
    hass: HomeAssistant
    region: str
    tz: tzinfo
    predictions: list[dict[str, Any]]
    confidence_scores: list[float]
    wind_model: NumpyRandomForest
    solar_model: NumpyRandomForest
    price_model: NumpyGradientBoosting
    wind_scaler: dict[str, float]
    solar_scaler: dict[str, float]
    price_scaler: dict[str, float]
    solar_scale: float
    _solar_scale_samples: int
    is_trained: bool
    training_samples: int
    last_trained_at: datetime | None
    _prices_updated_at: datetime | None
    _hpo_counter: int
    error_metrics: dict[int, dict[str, Any]]
    bias_correction: dict[int, float]
    volatility_mae: dict[int, float]
    learning_rate: float
    price_history: list[dict[str, Any]]
    max_history_days: int
    storage: LearningStorage
    lead_time_accuracy: dict[str, dict[str, float | int]]
    _prediction_insert_counter: int = 0

    # --- Cross-mixin methods, implemented in the sibling mixins ---
    def store_daily_prices(
        self, prices: Sequence[float | None], date: str | None = None
    ) -> bool:
        raise NotImplementedError

    def get_all_historical_prices(self) -> tuple[list[float], list[dict]]:
        raise NotImplementedError

    def store_prediction_for_learning(
        self,
        timestamp: str,
        predicted_price: float,
        confidence: float,
        forecast_temp: float | None = None,
        forecast_wind: float | None = None,
        forecast_cloud: float | None = None,
    ) -> None:
        raise NotImplementedError

    def _update_bias_correction(self, slot: int) -> None:
        raise NotImplementedError

    def record_lead_time_accuracy(
        self, predictions: list[dict[str, Any]], actual_price: float
    ) -> None:
        raise NotImplementedError

    def _train_models(
        self, historical_prices: Sequence[float | None], features: list[dict]
    ) -> None:
        raise NotImplementedError

    def _optimize_hyperparameters(self) -> dict | None:
        raise NotImplementedError

    def _restore_hpo_counter(self, data: dict[str, Any]) -> None:
        raise NotImplementedError
