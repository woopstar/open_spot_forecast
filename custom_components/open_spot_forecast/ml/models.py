"""Model training and prediction generation."""

import contextlib
import logging
from datetime import UTC, datetime, timedelta

import numpy as np

from homeassistant.util import dt as dt_util

from .base import PredictorBase
from .features import build_feature_vector
from .numpy_models import NumpyGradientBoosting

_LOGGER = logging.getLogger(__name__)


def create_price_model() -> NumpyGradientBoosting:
    """Return an untrained price model with the default production hyperparameters.

    Used by ``SpotPricePredictor`` and the dev backtest (``scripts/backtest.py``),
    so both always evaluate the same model configuration.
    """
    return NumpyGradientBoosting(n_estimators=200, learning_rate=0.1, random_state=42)


class ModelMixin(PredictorBase):
    """Training, prediction, and bias-correction methods.

    Designed to be mixed into SpotPricePredictor — all attributes
    referenced via self are provided by the owning class.
    """

    @staticmethod
    def _normalize_to_utc(iso_timestamp: str) -> str | None:
        """Convert an ISO timestamp with timezone to UTC format.

        Nordpool API returns UTC timestamps (e.g. 2026-07-08T12:00:00Z).
        Training timestamps may have local timezone offsets (e.g.
        2026-07-08T14:00:00+02:00). This normalizes both to UTC so
        lookups in the nordpool_prognoses table match reliably.
        """
        try:
            dt = datetime.fromisoformat(iso_timestamp)
            if dt.tzinfo is not None:
                dt = dt.astimezone(UTC)
            return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        except ValueError, TypeError:
            return None

    def _train_models(
        self, historical_prices: list[float], features: list[dict]
    ) -> None:
        """Train ML models on historical data.

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

            # Extract weather and price stats from provided features
            if features:
                sample_feature = features[0]
                wind_features = {
                    "wind_speed_mean": sample_feature.get("wind_speed_mean", 0),
                    "wind_power_estimate": sample_feature.get("wind_power_estimate", 0),
                    "wind_direction": sample_feature.get("wind_direction", 0),
                }
                solar_features = {
                    "solar_radiation_mean": sample_feature.get(
                        "solar_radiation_mean", 0
                    ),
                    "solar_power_estimate": sample_feature.get(
                        "solar_power_estimate", 0
                    ),
                }
                price_stats = {
                    "price_mean": float(np.mean(all_prices)),
                }
                temperature = sample_feature.get("temperature", 15.0)
            else:
                wind_features = {
                    "wind_speed_mean": 0,
                    "wind_power_estimate": 0,
                    "wind_direction": 0,
                }
                solar_features = {"solar_radiation_mean": 0, "solar_power_estimate": 0}
                price_stats = {"price_mean": float(np.mean(all_prices))}
                temperature = 15.0

            # Add weather and price stats to all historical features
            # Try to use stored historical weather and Nordpool prognoses for each slot
            for feature in all_features:
                start = feature.get("start", "")
                hist_weather = (
                    self.storage.find_weather_for_timestamp(start) if start else None
                )

                if hist_weather:
                    feature["temperature"] = (
                        hist_weather.get("temperature") or temperature
                    )
                    feature["wind_speed_mean"] = hist_weather.get(
                        "wind_speed"
                    ) or wind_features.get("wind_speed_mean", 0)
                    feature["wind_direction"] = hist_weather.get(
                        "wind_direction"
                    ) or wind_features.get("wind_direction", 0)
                    feature["cloud_coverage"] = hist_weather.get("cloud_coverage", 0)
                    feature["humidity"] = hist_weather.get("humidity", 50)
                    feature["solar_power_estimate"] = hist_weather.get(
                        "solar_power"
                    ) or solar_features.get("solar_power_estimate", 0)
                else:
                    feature.update(wind_features)
                    feature.update(solar_features)
                    feature["temperature"] = temperature
                feature.update(price_stats)

                # Look up stored Nordpool prognoses for this training slot
                if start:
                    utc_start = self._normalize_to_utc(start)
                    np_data = (
                        self.storage.find_nordpool_for_timestamp(utc_start)
                        if utc_start
                        else None
                    )
                    if np_data:
                        cons = np_data.get("consumption") or 0
                        sol = np_data.get("solar") or 0
                        woff = np_data.get("wind_offshore") or 0
                        won = np_data.get("wind_onshore") or 0
                        feature["consumption_forecast"] = cons
                        feature["solar_generation"] = sol
                        feature["wind_offshore"] = woff
                        feature["wind_onshore"] = won
                        feature["net_demand"] = cons - sol - woff - won
                        feature["wind_share"] = (woff + won) / cons if cons > 0 else 0

            # Prepare training data
            X_list: list[list[float]] = []
            y_list: list[float] = []

            for i, feature in enumerate(all_features):
                feature_vector = build_feature_vector(feature)
                X_list.append(feature_vector)
                y_list.append(all_prices[i])

            X = np.array(X_list)
            y = np.array(y_list)

            # Train/test split (80/20)
            split_idx = int(0.8 * len(X))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]

            # Train price model
            self.price_model.fit(X_train, y_train)

            # Evaluate
            y_pred = self.price_model.predict(X_test)
            mae = float(np.mean(np.abs(y_test - y_pred)))
            rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))

            _LOGGER.info(
                "ML model trained: MAE=%.2f, RMSE=%.2f, samples=%d, days=%d",
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

        Tests combinations of n_estimators and learning_rate using
        the stored price history. Best parameters are saved to the
        storage meta table. Returns the best param dict or None if
        insufficient data.

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

        best_params = {"n_estimators": 200, "learning_rate": 0.1}
        best_score = float("inf")

        param_grid = [(n, lr) for n in [100, 200, 300] for lr in [0.05, 0.1, 0.2]]

        for n_est, lr in param_grid:
            model = NumpyGradientBoosting(
                n_estimators=n_est, learning_rate=lr, random_state=42
            )
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                mae = float(np.mean(np.abs(y_val - y_pred)))
            except Exception:
                continue

            if mae < best_score:
                best_score = mae
                best_params = {"n_estimators": n_est, "learning_rate": lr}
                _LOGGER.debug(
                    "HPO candidate: n=%d lr=%.2f MAE=%.4f",
                    n_est,
                    lr,
                    mae,
                )

        _LOGGER.info(
            "Hyperparameter optimization complete: n_estimators=%d learning_rate=%.2f "
            "best_MAE=%.4f",
            best_params["n_estimators"],
            best_params["learning_rate"],
            best_score,
        )

        # Apply best params to the live model
        self.price_model = NumpyGradientBoosting(
            n_estimators=int(best_params["n_estimators"]),
            learning_rate=best_params["learning_rate"],
            random_state=42,
        )
        # The new model is unfitted until the caller retrains it
        self.is_trained = False

        # Persist best params
        with contextlib.suppress(Exception):
            self.storage.save_meta_dict(
                {
                    "hpo_n_estimators": str(best_params["n_estimators"]),
                    "hpo_learning_rate": str(best_params["learning_rate"]),
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
                "Sample feature values: hour=%s, wind=%s, solar=%s, price_mean=%s",
                sample_feature.get("hour"),
                sample_feature.get("wind_speed_mean"),
                sample_feature.get("solar_radiation_mean"),
                sample_feature.get("price_mean"),
            )

        for idx, feature in enumerate(features):
            feature_vector = build_feature_vector(feature)

            # Log first few feature vectors
            if idx < 3:
                _LOGGER.info(
                    "Feature vector %d (before sanitization): %s",
                    idx,
                    [
                        feature.get("hour", 0),
                        feature.get("day_of_week", 0),
                        feature.get("is_weekend", 0),
                        feature.get("hour_sin", 0),
                        feature.get("hour_cos", 0),
                        feature.get("wind_speed_mean", 0),
                        feature.get("wind_power_estimate", 0),
                        feature.get("solar_radiation_mean", 0),
                        feature.get("solar_power_estimate", 0),
                        feature.get("price_mean", 0),
                        feature.get("temperature", 15.0),
                    ],
                )
                _LOGGER.info(
                    "Feature vector %d (after sanitization): %s",
                    idx,
                    feature_vector,
                )

            # Predict price
            feature_array = np.array([feature_vector])
            price = float(self.price_model.predict(feature_array)[0])

            start = feature.get("start")

            # Log first few predictions
            if idx < 3:
                _LOGGER.info(
                    "Prediction %d: start=%s, raw_price=%.4f, feature_array_shape=%s",
                    idx,
                    start,
                    price,
                    feature_array.shape,
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
                    "price": max(0, price),
                    "confidence": confidence,
                    "model": "GradientBoosting",
                }
            )

            self.confidence_scores.append(confidence)

            # Store prediction for self-learning (with forecast weather)
            if start:
                self.store_prediction_for_learning(
                    start,
                    max(0, price),
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
        historical_prices: list[float],
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
        if not historical_prices:
            _LOGGER.warning("No historical prices for heuristic predictions")
            return

        # Use historical average with time-of-day adjustment
        avg_price = float(np.mean(historical_prices))
        hourly_pattern = self._extract_hourly_pattern(historical_prices)

        _LOGGER.info(
            "Heuristic: avg_price=%.4f, hourly_pattern_length=%d",
            avg_price,
            len(hourly_pattern),
        )

        now = dt_util.utcnow()

        # Start from the next whole hour
        next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)

        # If we have known/confirmed prices, start after those end
        if known_data_end_time is not None:
            if known_data_end_time.minute > 0 or known_data_end_time.second > 0:
                ceil_end = known_data_end_time.replace(
                    minute=0, second=0, microsecond=0
                ) + timedelta(hours=1)
            else:
                ceil_end = known_data_end_time.replace(
                    minute=0, second=0, microsecond=0
                )
            start_time = max(next_hour, ceil_end)
        else:
            start_time = next_hour

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
                        "price": max(0, price),
                        "confidence": confidence,
                        "model": "heuristic",
                    }
                )

                self.confidence_scores.append(confidence)

                # Store prediction for self-learning
                self.store_prediction_for_learning(start_str, max(0, price), confidence)

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

        Uses the slot's MAE relative to its mean actual price when
        sufficient learning data exists, then adjusts for volatility
        (per-slot EMA of MAE) and weather forecast accuracy.
        Falls back to a heuristic based on feature quality and forecast
        distance.
        """
        hour = feature.get("hour", 0)
        minute = feature.get("minute", 0)
        slot = hour * 4 + minute // 15

        # Use learned metrics if we have enough data for this slot
        if slot in self.error_metrics:
            metrics = self.error_metrics[slot]
            if metrics["count"] >= 5:
                mae = float(np.mean(metrics["abs_errors"]))
                mean_actual = float(np.mean(metrics["actuals"]))
                if mean_actual > 0:
                    error_ratio = mae / mean_actual
                    learned_confidence = max(0.1, 1.0 - error_ratio)

                    # --- Volatility adjustment ---
                    # Higher per-slot volatility → lower confidence
                    vol = self.volatility_mae.get(slot)
                    if vol is not None and vol > 0 and mean_actual > 0:
                        vol_ratio = vol / mean_actual
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

        if feature.get("wind_speed_mean", 0) == 0:
            confidence -= 0.2
        if feature.get("solar_radiation_mean", 0) == 0:
            confidence -= 0.1
        if feature.get("is_weekend", 0) == 1:
            confidence -= 0.05

        start = feature.get("start")
        if start:
            try:
                dt = datetime.fromisoformat(start)
                days_ahead = (dt - datetime.now()).days
                confidence -= days_ahead * 0.05
            except ValueError, TypeError:
                pass

        return max(0.3, min(1.0, confidence))

    def apply_bias_correction(self, predicted_price: float, slot: int) -> float:
        """Apply learned bias correction to a prediction.

        Args:
            predicted_price: Raw predicted price
            slot: 15-minute slot index (0-95)

        Returns:
            Corrected price after applying learned bias adjustment
        """
        if slot in self.bias_correction:
            correction = self.bias_correction[slot]
            corrected_price = predicted_price * correction

            _LOGGER.debug(
                "Applied bias correction for slot %d: %.2f -> %.2f (factor=%.3f)",
                slot,
                predicted_price,
                corrected_price,
                correction,
            )

            return max(0, corrected_price)

        return predicted_price

    def _update_bias_correction(self, slot: int) -> None:
        """Update bias correction factor for a specific 15-minute slot.

        Uses exponential moving average to adaptively learn systematic
        prediction biases for each 15-minute slot of the day.
        """
        try:
            metrics = self.error_metrics.get(slot)
            if not metrics or metrics["count"] < 3:
                return

            mean_error = float(np.mean(metrics["errors"]))
            mean_actual = float(np.mean(metrics["actuals"]))

            if mean_actual > 0:
                bias_ratio = 1.0 - (mean_error / mean_actual)

                if slot in self.bias_correction:
                    old_correction = self.bias_correction[slot]
                    self.bias_correction[slot] = (
                        1 - self.learning_rate
                    ) * old_correction + self.learning_rate * bias_ratio
                else:
                    self.bias_correction[slot] = bias_ratio

                _LOGGER.debug(
                    "Updated bias correction for slot %d: %.3f (mean_error=%.2f)",
                    slot,
                    self.bias_correction[slot],
                    mean_error,
                )

        except Exception as err:
            _LOGGER.error("Error updating bias correction for slot %d: %s", slot, err)
