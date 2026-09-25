"""Model training and prediction generation."""

import contextlib
import logging
from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from homeassistant.util import dt as dt_util

from ..price_series import known_prices
from ..time_slots import first_prediction_slot
from .base import PredictorBase
from .features import build_feature_vector
from .gbm import NumpyGradientBoosting

_LOGGER = logging.getLogger(__name__)

# Production hyperparameters, chosen with the backtest (docs/ml_documentation.md);
# hyperparameter optimization may replace the first three per installation
DEFAULT_N_ESTIMATORS = 200
DEFAULT_LEARNING_RATE = 0.05
DEFAULT_MAX_DEPTH = 3
# About one day of 15-minute slots: a leaf never describes a single day's noise
MIN_SAMPLES_LEAF = 100

# Hyperparameter search grid. n_estimators is scored from one fit per
# (learning_rate, max_depth) pair with staged predictions, so the grid costs
# len(HPO_LEARNING_RATES) * len(HPO_MAX_DEPTHS) fits, not the full product.
HPO_N_ESTIMATORS = (100, 200, 300)
HPO_LEARNING_RATES = (0.05, 0.1, 0.2)
HPO_MAX_DEPTHS = (2, 3, 4, 6)


def create_price_model(
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> NumpyGradientBoosting:
    """Return an untrained price model; defaults are the production hyperparameters.

    Used by ``SpotPricePredictor``, hyperparameter search and the dev backtest
    (``scripts/backtest.py``), so all of them evaluate the same model
    configuration. Leaf count and L2 regularization keep their
    ``NumpyGradientBoosting`` defaults.
    """
    return NumpyGradientBoosting(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=42,
    )


class ModelMixin(PredictorBase):
    """Training, prediction, and bias-correction methods.

    Designed to be mixed into SpotPricePredictor — all attributes
    referenced via self are provided by the owning class.
    """

    def _train_models(self) -> None:
        """Train the price model on all stored price history.

        Training rows come from ``get_all_historical_prices`` (stored
        actuals and prognoses), never from the current forecast's features.
        Today's prices must already be in price_history (see
        RetrainMixin.record_training_prices).
        """
        try:
            # Get ALL historical prices and features (multi-day training)
            all_prices, all_features = self.get_all_historical_prices()

            if len(all_prices) < 24:
                _LOGGER.debug(
                    "Not enough historical data to train models (need 24, have %d)",
                    len(all_prices),
                )
                return

            _LOGGER.info(
                "Training on %d samples from %d days of history",
                len(all_prices),
                len(self.price_history),
            )

            # Prepare training data
            X_list: list[list[float]] = []
            y_list: list[float] = []

            for i, feature in enumerate(all_features):
                feature_vector = build_feature_vector(feature)
                X_list.append(feature_vector)
                y_list.append(all_prices[i])

            X = np.array(X_list)
            y = np.array(y_list)

            # Chronological train/test split (80/20), used only to measure
            # holdout error with a copy of the model on the oldest 80 %
            split_idx = int(0.8 * len(X))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]

            holdout_model = NumpyGradientBoosting(**self.price_model.get_params())
            holdout_model.fit(X_train, y_train)

            # Evaluate
            y_pred = holdout_model.predict(X_test)
            mae = float(np.mean(np.abs(y_test - y_pred)))
            rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))

            # The live model trains on all rows, so the most recent days
            # (closest to what is being predicted) are part of it
            self.price_model.fit(X, y)

            _LOGGER.info(
                "ML model trained: holdout MAE=%.2f, RMSE=%.2f, samples=%d, days=%d",
                mae,
                rmse,
                len(all_prices),
                len(self.price_history),
            )

            self.is_trained = True
            self.training_samples = len(all_prices)

        except Exception as err:
            _LOGGER.error("Error training ML models: %s", err, exc_info=True)
            self.is_trained = False

    def _optimize_hyperparameters(self) -> dict | None:
        """Run grid search for best GradientBoosting hyperparameters.

        Tests every combination of n_estimators, learning_rate and max_depth
        in the HPO_* grids using the stored price history. Best parameters are
        saved to the storage meta table. Returns the best param dict or None
        if insufficient data.

        Called by RetrainMixin.retrain once per HPO_INTERVAL_DAYS new days
        of price data.
        """
        if len(self.price_history) < 7:
            _LOGGER.debug("Not enough history for hyperparameter opt (need 7 days)")
            return None

        _LOGGER.info("Starting hyperparameter optimization...")

        # Build training data from all history
        all_prices, all_features = self.get_all_historical_prices()
        if len(all_prices) < 168:  # min 7 days * 24 hours
            return None

        X_list: list[list[float]] = []
        y_list: list[float] = []
        for i, feature in enumerate(all_features):
            feature_vector = build_feature_vector(feature)
            X_list.append(feature_vector)
            y_list.append(all_prices[i])

        X = np.array(X_list)
        y = np.array(y_list)

        # 80/20 train/validation split
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        best_params: dict[str, Any] = {
            "n_estimators": DEFAULT_N_ESTIMATORS,
            "learning_rate": DEFAULT_LEARNING_RATE,
            "max_depth": DEFAULT_MAX_DEPTH,
        }
        best_score = float("inf")

        for lr in HPO_LEARNING_RATES:
            for depth in HPO_MAX_DEPTHS:
                model = create_price_model(
                    n_estimators=max(HPO_N_ESTIMATORS),
                    learning_rate=lr,
                    max_depth=depth,
                )
                try:
                    model.fit(X_train, y_train)
                    staged = model.staged_predict(X_val)
                    for n_trees, y_pred in enumerate(staged, start=1):
                        if n_trees not in HPO_N_ESTIMATORS:
                            continue
                        mae = float(np.mean(np.abs(y_val - y_pred)))
                        if mae < best_score:
                            best_score = mae
                            best_params = {
                                "n_estimators": n_trees,
                                "learning_rate": lr,
                                "max_depth": depth,
                            }
                            _LOGGER.debug(
                                "HPO candidate: n=%d lr=%.2f depth=%d MAE=%.4f",
                                n_trees,
                                lr,
                                depth,
                                mae,
                            )
                except Exception:
                    continue

        _LOGGER.info(
            "Hyperparameter optimization complete: n_estimators=%d learning_rate=%.2f "
            "max_depth=%d best_MAE=%.4f",
            best_params["n_estimators"],
            best_params["learning_rate"],
            best_params["max_depth"],
            best_score,
        )

        # Apply best params to the live model
        self.price_model = create_price_model(**best_params)
        # The new model is unfitted until the caller retrains it
        self.is_trained = False

        # Persist best params
        with contextlib.suppress(Exception):
            self.storage.save_meta_dict(
                {
                    "hpo_n_estimators": str(best_params["n_estimators"]),
                    "hpo_learning_rate": str(best_params["learning_rate"]),
                    "hpo_max_depth": str(best_params["max_depth"]),
                    "hpo_best_mae": str(best_score),
                }
            )

        return best_params

    def _generate_predictions(
        self, features: list[dict], forecast_days: int, interval_minutes: int
    ) -> None:
        """Generate predictions using trained models."""
        self.predictions = []
        self.confidence_scores = []

        _LOGGER.info("Generating predictions for %d features", len(features))

        # Log feature statistics
        if features:
            sample_feature = features[0]
            _LOGGER.info(
                "Sample feature keys: %s",
                list(sample_feature.keys())[:10],
            )
            _LOGGER.info(
                "Sample feature values: hour=%s, wind=%s, solar=%s, consumption=%s",
                sample_feature.get("hour"),
                sample_feature.get("wind_speed_mean"),
                sample_feature.get("solar_generation"),
                sample_feature.get("consumption_forecast"),
            )

        # One batch call: walking every tree once per slot costs ~100x more
        feature_vectors = [build_feature_vector(feature) for feature in features]
        raw_prices = (
            self.price_model.predict(np.array(feature_vectors))
            if feature_vectors
            else np.empty(0)
        )

        for idx, feature in enumerate(features):
            feature_vector = feature_vectors[idx]

            # Log first few feature vectors
            if idx < 3:
                _LOGGER.info(
                    "Feature vector %d (NaN = unknown input): %s",
                    idx,
                    feature_vector,
                )

            price = float(raw_prices[idx])

            start = feature.get("start")

            # Log first few predictions
            if idx < 3:
                _LOGGER.info(
                    "Prediction %d: start=%s, raw_price=%.4f",
                    idx,
                    start,
                    price,
                )

            # Apply bias correction if available (keyed by 15-min slot 0-95)
            hour = feature.get("hour", 0)
            minute = feature.get("minute", 0)
            slot = hour * 4 + minute // 15
            price = self.apply_bias_correction(price, slot)
            confidence = round(self._estimate_confidence(feature), 2)

            self.predictions.append(
                {
                    "start": start,
                    "end": feature.get("end"),
                    "price": price,
                    "confidence": confidence,
                    "model": "GradientBoosting",
                }
            )

            self.confidence_scores.append(confidence)

            # Store prediction for self-learning (with forecast weather)
            if start:
                self.store_prediction_for_learning(
                    start,
                    price,
                    confidence,
                    forecast_temp=feature.get("temperature"),
                    forecast_wind=feature.get("wind_speed_mean"),
                    forecast_cloud=feature.get("cloud_coverage"),
                )

        _LOGGER.info(
            "Generated %d predictions, price range: %.4f - %.4f",
            len(self.predictions),
            min(p["price"] for p in self.predictions) if self.predictions else 0,
            max(p["price"] for p in self.predictions) if self.predictions else 0,
        )

        # Force WAL checkpoint so predictions survive process restart
        self.storage.checkpoint()
        pred_count = self.storage.count_predictions()
        _LOGGER.info(
            "Prediction storage: %d total predictions in database after this run",
            pred_count,
        )

        # Log first 5 predictions with timestamps for debugging
        if self.predictions:
            _LOGGER.info("First 5 predictions:")
            for i, pred in enumerate(self.predictions[:5]):
                _LOGGER.info(
                    "  [%d] start=%s, price=%.4f, confidence=%.2f",
                    i,
                    pred.get("start"),
                    pred.get("price", 0),
                    pred.get("confidence", 0),
                )

    def _generate_heuristic_predictions(
        self,
        historical_prices: Sequence[float | None],
        forecast_days: int,
        interval_minutes: int,
        known_data_end_time: datetime | None = None,
    ) -> None:
        """Generate predictions using simple heuristics (fallback)."""
        self.predictions = []
        self.confidence_scores = []

        _LOGGER.info(
            "Generating heuristic predictions with %d historical prices",
            len(historical_prices),
        )
        # Slots missing in the source (None) carry no price
        known = known_prices(historical_prices)
        if not known:
            _LOGGER.warning("No historical prices for heuristic predictions")
            return

        # Use historical average with time-of-day adjustment
        avg_price = float(np.mean(known))
        hourly_pattern = self._extract_hourly_pattern(known)

        _LOGGER.info(
            "Heuristic: avg_price=%.4f, hourly_pattern_length=%d",
            avg_price,
            len(hourly_pattern),
        )

        start_time = first_prediction_slot(
            dt_util.utcnow(), known_data_end_time, interval_minutes
        )

        intervals_per_day = (24 * 60) // interval_minutes

        for day in range(forecast_days):
            for interval in range(intervals_per_day):
                dt = start_time + timedelta(
                    days=day, minutes=interval * interval_minutes
                )
                dt_local = dt_util.as_local(dt)

                hour = dt_local.hour

                # Apply hourly pattern
                if hour < len(hourly_pattern):
                    price = avg_price * hourly_pattern[hour]
                else:
                    price = avg_price

                # Reduce confidence for further predictions
                confidence = max(0.3, 1.0 - (day * 0.1))

                dt_end = dt + timedelta(minutes=interval_minutes)
                dt_end_local = dt_util.as_local(dt_end)
                start_str = dt_local.isoformat()
                end_str = dt_end_local.isoformat()

                self.predictions.append(
                    {
                        "start": start_str,
                        "end": end_str,
                        "price": price,
                        "confidence": confidence,
                        "model": "heuristic",
                    }
                )

                self.confidence_scores.append(confidence)

                # Store prediction for self-learning
                self.store_prediction_for_learning(start_str, price, confidence)

        _LOGGER.info(
            "Generated %d heuristic predictions, price range: %.4f - %.4f",
            len(self.predictions),
            min(p["price"] for p in self.predictions) if self.predictions else 0,
            max(p["price"] for p in self.predictions) if self.predictions else 0,
        )

    def _extract_hourly_pattern(self, prices: list[float]) -> list[float]:
        """Extract hourly price pattern from historical data."""
        if len(prices) < 24:
            return [1.0] * 24

        # Reshape to 24-hour periods
        n_days = len(prices) // 24
        if n_days == 0:
            return [1.0] * 24

        hourly_avg = []
        for hour in range(24):
            hour_prices = [prices[day * 24 + hour] for day in range(n_days)]
            hourly_avg.append(float(np.mean(hour_prices)))

        # Normalize
        overall_avg = float(np.mean(hourly_avg))
        if overall_avg > 0:
            pattern = [h / overall_avg for h in hourly_avg]
        else:
            pattern = [1.0] * 24

        return pattern

    def _estimate_confidence(self, feature: dict) -> float:
        """Estimate prediction confidence using learned error metrics.

        Uses the slot's MAE relative to its price scale, the mean absolute
        actual price, when sufficient learning data exists, then adjusts for
        volatility (per-slot EMA of MAE) and weather forecast accuracy. The
        scale is positive for negative prices too, so slots that clear at or
        below zero still get a learned confidence. Falls back to a heuristic
        based on feature quality and forecast distance.
        """
        hour = feature.get("hour", 0)
        minute = feature.get("minute", 0)
        slot = hour * 4 + minute // 15

        # Use learned metrics if we have enough data for this slot
        if slot in self.error_metrics:
            metrics = self.error_metrics[slot]
            if metrics["count"] >= 5:
                mae = float(np.mean(metrics["abs_errors"]))
                price_scale = float(np.mean(np.abs(metrics["actuals"])))
                if price_scale > 1e-9:
                    error_ratio = mae / price_scale
                    learned_confidence = max(0.1, 1.0 - error_ratio)

                    # --- Volatility adjustment ---
                    # Higher per-slot volatility → lower confidence
                    vol = self.volatility_mae.get(slot)
                    if vol is not None and vol > 0:
                        vol_ratio = vol / price_scale
                        # Reduce confidence proportional to volatility
                        # Volatility of 50% of mean → ~0.15 reduction
                        learned_confidence -= min(0.25, vol_ratio * 0.3)

                    # Penalize for inaccurate weather forecasts
                    fc_temp_errs = metrics.get("forecast_temp_errors", [])
                    fc_wind_errs = metrics.get("forecast_wind_errors", [])
                    if fc_temp_errs:
                        avg_temp_err = float(np.mean(fc_temp_errs))
                        # >2°C forecast error → reduce confidence
                        learned_confidence -= min(0.15, avg_temp_err * 0.05)
                    if fc_wind_errs:
                        avg_wind_err = float(np.mean(fc_wind_errs))
                        # >3 m/s forecast error → reduce confidence
                        learned_confidence -= min(0.15, avg_wind_err * 0.03)

                    return round(max(0.1, learned_confidence), 2)

        # Fallback: heuristic based on feature quality and forecast distance
        confidence = 0.8  # Base confidence

        # No weather forecast or market prognosis for the slot
        if feature.get("wind_speed_mean") is None:
            confidence -= 0.2
        if feature.get("solar_generation") is None:
            confidence -= 0.1
        if feature.get("is_weekend", 0) == 1:
            confidence -= 0.05

        start = feature.get("start")
        if start:
            try:
                dt = datetime.fromisoformat(start)
                # Clamp: the current slot started in the past (.days == -1)
                days_ahead = max(0, (dt - dt_util.now()).days)
                confidence -= days_ahead * 0.05
            except ValueError, TypeError:
                pass

        return max(0.3, min(1.0, confidence))

    def apply_bias_correction(self, predicted_price: float, slot: int) -> float:
        """Subtract the slot's learned bias offset from a prediction.

        The correction is additive, so it works the same for positive,
        zero and negative prices and never flips a prediction's sign by
        scaling it. The result is not clamped: prices can be negative.

        Args:
            predicted_price: Raw model prediction
            slot: 15-minute slot index (0-95)

        Returns:
            ``predicted_price - offset[slot]`` (unchanged without an offset)
        """
        offset = self.bias_correction.get(slot)
        if offset is None:
            return predicted_price

        corrected_price = predicted_price - offset
        _LOGGER.debug(
            "Applied bias correction for slot %d: %.4f -> %.4f (offset=%.4f)",
            slot,
            predicted_price,
            corrected_price,
            offset,
        )
        return corrected_price

    def _update_bias_correction(self, slot: int) -> None:
        """Update the additive bias offset of one 15-minute slot.

        ``mean_error`` (predicted - actual) is measured on stored predictions,
        which already had the current offset subtracted, so the raw model's
        bias is ``offset + mean_error``. The offset is an EMA of that bias:

            offset = 0.9 * offset + 0.1 * (offset + mean_error)

        Feeding the EMA ``mean_error`` alone would settle at half the bias.
        The first update sets the offset to ``mean_error``. No division by a
        price, so the offset stays bounded when prices are zero or negative.
        """
        try:
            metrics = self.error_metrics.get(slot)
            if not metrics or metrics["count"] < 3:
                return

            mean_error = float(np.mean(metrics["errors"]))
            old_offset = self.bias_correction.get(slot)
            if old_offset is None:
                self.bias_correction[slot] = mean_error
            else:
                raw_bias = old_offset + mean_error
                self.bias_correction[slot] = (
                    1 - self.learning_rate
                ) * old_offset + self.learning_rate * raw_bias

            _LOGGER.debug(
                "Updated bias offset for slot %d: %.4f (mean_error=%.4f)",
                slot,
                self.bias_correction[slot],
                mean_error,
            )

        except Exception as err:
            _LOGGER.error("Error updating bias correction for slot %d: %s", slot, err)
