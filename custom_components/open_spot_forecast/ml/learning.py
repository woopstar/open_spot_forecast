"""Self-learning, historical storage, and persistence."""

import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import numpy as np

from homeassistant.util import dt as dt_util

from ..price_series import is_invalid_price_series
from ..time_slots import slot_start_in_day
from .base import PredictorBase
from .features import build_feature_row
from .training_inputs import TrainingInputs

_LOGGER = logging.getLogger(__name__)


def percent_error(error: float, actual_price: float) -> float:
    """Return ``error`` as a percentage of the actual price's magnitude.

    Relative to ``|actual|`` so negative prices get a meaningful percentage;
    0.0 for a zero price, where a percentage is undefined.
    """
    return error / abs(actual_price) * 100 if abs(actual_price) > 1e-9 else 0.0


def predictions_at_instant(
    predictions: list[dict[str, Any]], moment: datetime
) -> list[dict[str, Any]]:
    """Keep the predictions whose ``start`` is the same instant as ``moment``.

    Storage matches predictions on local date, hour and minute, which is
    ambiguous in the repeated hour of a DST fall-back (02:15 happens twice).
    Comparing UTC instants keeps only the right pass. A naive ``moment`` or
    ``start`` carries no offset to compare, so nothing is filtered on it.

    Args:
        predictions: Rows from ``find_predictions_for_timestamp``.
        moment: Start of the slot being learned from.

    Returns:
        The predictions for that exact slot.
    """
    if moment.tzinfo is None:
        return predictions
    # Compare in UTC: PEP 495 makes a fold=1 time unequal to other zones
    target = moment.astimezone(UTC)
    kept = []
    for prediction in predictions:
        start = dt_util.parse_datetime(str(prediction.get("start", "")))
        if start is None or start.tzinfo is None or start.astimezone(UTC) == target:
            kept.append(prediction)
    return kept


class LearningMixin(PredictorBase):
    """Historical price storage, self-learning, and persistence methods.

    Designed to be mixed into SpotPricePredictor — all attributes
    referenced via self are provided by the owning class.
    """

    def store_daily_prices(
        self, prices: Sequence[float | None], date: str | None = None
    ) -> bool:
        """Store today's prices for historical training data.

        An invalid day (all zero, or with missing values) is not stored, so
        good prices already stored for that date are kept.

        Args:
            prices: List of prices for the day (96 intervals for 15-min data)
            date: Date string (YYYY-MM-DD), defaults to today

        Returns:
            True if the prices were stored, False if they were rejected.
        """
        if date is None:
            date = dt_util.now().strftime("%Y-%m-%d")

        if is_invalid_price_series(prices):
            _LOGGER.warning(
                "Not storing prices for %s: all zero or with missing values", date
            )
            return False

        # Check if we already have this date
        for entry in self.price_history:
            if entry.get("date") == date:
                entry["prices"] = list(prices)
                _LOGGER.debug(
                    "Updated price history for %s (%d prices)", date, len(prices)
                )
                return True

        # Add new entry
        self.price_history.append(
            {
                "date": date,
                "prices": list(prices),
            }
        )

        # Keep the newest max_history_days. Days can arrive out of order (a
        # backfill of older days, #27), so sort by date before trimming
        self.price_history.sort(key=lambda entry: str(entry.get("date", "")))
        if len(self.price_history) > self.max_history_days:
            self.price_history = self.price_history[-self.max_history_days :]

        _LOGGER.info(
            "Stored price history for %s (%d prices, total days: %d)",
            date,
            len(prices),
            len(self.price_history),
        )
        return True

    def get_all_historical_prices(self) -> tuple[list[float], list[dict]]:
        """Get all historical prices and their training feature rows.

        Every row comes from ``build_feature_row``, the function prediction
        rows come from, with the slot's stored weather snapshot and Nordpool
        prognoses as inputs (``TrainingInputs``). Inputs that were not stored
        for a slot stay unknown (NaN for the model).

        Returns:
            Tuple of (all_prices, all_features) where:
            - all_prices: Flat list of all prices from all days
            - all_features: List of feature dicts for each price
        """
        all_prices = []
        all_features = []
        inputs = TrainingInputs(
            self.storage.load_weather_history(),
            self.storage.load_nordpool_history(),
            self.tz,
        )

        for entry in self.price_history:
            date_str = entry.get("date")
            prices = entry.get("prices", [])

            if not date_str or not prices:
                continue

            # Parse date
            try:
                date = datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                continue

            # Generate features for this day. Slot starts step on the UTC
            # timeline from local midnight, so a 92- or 100-slot DST day gets
            # the real wall-clock time and offset for every slot
            for interval, price in enumerate(prices):
                # A slot missing in the source (None) is not a training row;
                # the slots after it keep their own times
                if price is None:
                    continue
                start = slot_start_in_day(date.date(), interval, self.tz)
                all_prices.append(price)
                all_features.append(build_feature_row(start, inputs.for_slot(start)))

        _LOGGER.info(
            "Retrieved %d historical prices from %d days",
            len(all_prices),
            len(self.price_history),
        )
        return all_prices, all_features

    def store_prediction_for_learning(
        self,
        timestamp: str,
        predicted_price: float,
        confidence: float,
        forecast_temp: float | None = None,
        forecast_wind: float | None = None,
        forecast_cloud: float | None = None,
    ) -> None:
        """Store a prediction for later comparison with actual price.

        Writes directly to the SQLite database via storage layer.
        Prunes predictions older than max_history_days.

        Args:
            timestamp: ISO format timestamp of the prediction
            predicted_price: The predicted price
            confidence: Confidence score of the prediction
            forecast_temp: Temperature forecast used for this slot
            forecast_wind: Wind speed forecast used for this slot
            forecast_cloud: Cloud coverage forecast used for this slot
        """
        try:
            dt = datetime.fromisoformat(timestamp)
            stored_at = dt_util.now().isoformat()

            self.storage.insert_prediction(
                start=timestamp,
                price=predicted_price,
                confidence=confidence,
                hour=dt.hour,
                minute=dt.minute,
                stored_at=stored_at,
                forecast_temp=forecast_temp,
                forecast_wind=forecast_wind,
                forecast_cloud=forecast_cloud,
            )

            # Prune old entries periodically (every 100 inserts to avoid
            # doing it on every single prediction store)
            self._prediction_insert_counter = (
                getattr(self, "_prediction_insert_counter", 0) + 1
            )
            if self._prediction_insert_counter >= 100:
                deleted = self.storage.delete_old_predictions(self.max_history_days)
                self._prediction_insert_counter = 0
                if deleted > 0:
                    _LOGGER.debug(
                        "Pruned %d old predictions (older than %d days)",
                        deleted,
                        self.max_history_days,
                    )

            _LOGGER.debug(
                "Stored prediction for learning: %s (hour %d)", timestamp, dt.hour
            )

        except Exception as err:
            _LOGGER.error("Error storing prediction for learning: %s", err)

    def learn_from_actual_price(self, timestamp: str, actual_price: float) -> bool:
        """Compare prediction with actual price and update error metrics.

        This is the core of the self-learning loop. It:
        1. Finds the prediction made for this timestamp (from SQLite)
        2. Calculates the error (predicted vs actual)
        3. Updates the additive bias offset for this 15-min slot
        4. Adapts the model based on recent errors
        5. Records each error in its lead-time bucket

        Args:
            timestamp: ISO format timestamp of the actual price
            actual_price: The actual observed price

        Returns:
            True if a matching prediction was found and learning updated
        """
        try:
            # Find predictions for this timestamp
            dt = datetime.fromisoformat(timestamp)
            target_date = dt.strftime("%Y-%m-%d")
            target_hour = dt.hour
            target_minute = dt.minute
            target_slot = target_hour * 4 + target_minute // 15  # 0-95

            # Query SQLite for matching predictions
            matching_predictions = predictions_at_instant(
                self.storage.find_predictions_for_timestamp(
                    target_date, target_hour, target_minute
                ),
                dt,
            )

            if not matching_predictions:
                _LOGGER.info(
                    "No prediction found for %s (date=%s slot=%d)",
                    timestamp,
                    target_date,
                    target_slot,
                )
                return False

            # Learn from ALL matching predictions (multiple forecast runs
            # for the same timestamp = more training data)
            if target_slot not in self.error_metrics:
                self.error_metrics[target_slot] = {
                    "errors": [],
                    "abs_errors": [],
                    "pct_errors": [],
                    "predictions": [],
                    "actuals": [],
                    "count": 0,
                }
            metrics = self.error_metrics[target_slot]

            for prediction in matching_predictions:
                predicted_price = prediction.get("price", 0)

                # Calculate error for this prediction
                error = predicted_price - actual_price
                abs_error = abs(error)
                pct_error = percent_error(error, actual_price)

                metrics["errors"].append(error)
                metrics["abs_errors"].append(abs_error)
                metrics["pct_errors"].append(pct_error)
                metrics["predictions"].append(predicted_price)
                metrics["actuals"].append(actual_price)
                metrics["count"] += 1

            # Keep only recent errors (last 100 per slot)
            max_samples = 100
            for key in ["errors", "abs_errors", "pct_errors", "predictions", "actuals"]:
                if len(metrics[key]) > max_samples:
                    metrics[key] = metrics[key][-max_samples:]

            # Update bias correction for this 15-min slot
            self._update_bias_correction(target_slot)

            # --- Forecast accuracy: compare stored forecast vs actual weather ---
            actual_weather = self.storage.find_weather_for_timestamp(timestamp)
            forecast_errors = []
            for p in matching_predictions:
                fc_temp = p.get("forecast_temp")
                fc_wind = p.get("forecast_wind")
                if actual_weather and fc_temp is not None:
                    temp_err = abs(fc_temp - actual_weather.get("temperature", fc_temp))
                    wind_err = (
                        abs(fc_wind - actual_weather.get("wind_speed", fc_wind))
                        if fc_wind is not None
                        else 0
                    )
                    forecast_errors.append((temp_err, wind_err))

            if forecast_errors:
                avg_temp_err = float(np.mean([e[0] for e in forecast_errors]))
                avg_wind_err = float(np.mean([e[1] for e in forecast_errors]))
                _LOGGER.debug(
                    "Forecast accuracy for slot %d: temp_error=%.1f°C, wind_error=%.1f m/s",
                    target_slot,
                    avg_temp_err,
                    avg_wind_err,
                )
                # Store forecast errors for confidence calculation
                if "forecast_temp_errors" not in self.error_metrics[target_slot]:
                    self.error_metrics[target_slot]["forecast_temp_errors"] = []
                    self.error_metrics[target_slot]["forecast_wind_errors"] = []
                self.error_metrics[target_slot]["forecast_temp_errors"].append(
                    avg_temp_err
                )
                self.error_metrics[target_slot]["forecast_wind_errors"].append(
                    avg_wind_err
                )
                # Keep last 50
                for k in ["forecast_temp_errors", "forecast_wind_errors"]:
                    if len(self.error_metrics[target_slot][k]) > 50:
                        self.error_metrics[target_slot][k] = self.error_metrics[
                            target_slot
                        ][k][-50:]

            # Remove ALL matched predictions from SQLite (by id)
            for p in matching_predictions:
                self.storage.remove_prediction(p["id"])

            self.record_lead_time_accuracy(matching_predictions, actual_price)

            # --- Update per-slot volatility (EMA of MAE) ---
            mae = float(np.mean(metrics["abs_errors"]))
            old_vol = self.volatility_mae.get(target_slot, 0.0)
            alpha = 0.1  # EMA smoothing factor
            if old_vol > 0:
                self.volatility_mae[target_slot] = (1 - alpha) * old_vol + alpha * mae
            else:
                # First measurement: initialize with actual MAE
                self.volatility_mae[target_slot] = mae

            bias = float(np.mean(metrics["errors"]))
            latest_error = matching_predictions[-1]["price"] - actual_price

            _LOGGER.info(
                "Self-learning update (slot %d, %02d:%02d): %d predictions matched, "
                "latest_error=%.2f, MAE=%.2f, bias=%.2f, samples=%d, volatility=%.4f",
                target_slot,
                target_hour,
                target_minute,
                len(matching_predictions),
                latest_error,
                mae,
                bias,
                metrics["count"],
                self.volatility_mae.get(target_slot, 0),
            )

            return True

        except Exception as err:
            _LOGGER.error("Error in self-learning loop: %s", err)
            return False

    def get_learning_metrics(self) -> dict[str, Any]:
        """Get comprehensive learning and error metrics.

        Returns:
            Dictionary with learning statistics and error metrics, plus the
            latest training's ``holdout_mae``, ``holdout_rmse`` and
            ``holdout_trained_at`` (None before a successful training)
        """
        return self._comparison_metrics() | self.holdout_metrics()

    def _comparison_metrics(self) -> dict[str, Any]:
        """Return the self-learning metrics of predictions compared to actuals."""
        pending = self.storage.count_predictions()

        if not self.error_metrics:
            if pending > 0:
                return {
                    "status": "collecting",
                    "is_learning": False,
                    "total_samples": 0,
                    "pending_predictions": pending,
                    "message": f"Collecting data — {pending} predictions waiting to be compared (needs ~24h)",
                }
            return {
                "status": "idle",
                "is_learning": False,
                "total_samples": 0,
                "pending_predictions": 0,
                "message": "No data yet — predictions will be compared after 24h",
            }

        # Aggregate metrics across all slots
        all_errors = []
        all_abs_errors = []
        all_pct_errors = []
        total_samples = 0

        for __, metrics in self.error_metrics.items():
            all_errors.extend(metrics["errors"])
            all_abs_errors.extend(metrics["abs_errors"])
            all_pct_errors.extend(metrics["pct_errors"])
            total_samples += metrics["count"]

        if total_samples == 0:
            return {
                "status": "collecting",
                "is_learning": False,
                "total_samples": 0,
                "pending_predictions": pending,
                "message": f"Collecting data — {pending} predictions waiting to be compared",
            }

        # Calculate overall metrics
        mae = float(np.mean(all_abs_errors))
        rmse = float(np.sqrt(np.mean(np.array(all_errors) ** 2)))
        mean_bias = float(np.mean(all_errors))
        mean_pct_error = float(np.mean(all_pct_errors))

        # Calculate per-slot metrics (0-95 = 15-min intervals)
        slot_metrics = {}
        for slot, metrics in sorted(self.error_metrics.items()):
            if metrics["count"] > 0:
                hour = slot // 4
                minute = (slot % 4) * 15
                slot_metrics[str(slot)] = {
                    "hour": hour,
                    "minute": minute,
                    "mae": float(np.mean(metrics["abs_errors"])),
                    "bias": float(np.mean(metrics["errors"])),
                    "samples": metrics["count"],
                    "bias_correction": self.bias_correction.get(slot, 0.0),
                    "volatility": self.volatility_mae.get(slot),
                }

        # Calculate confidence in learning (based on sample size)
        learning_confidence = min(1.0, total_samples / 100)

        return {
            "status": "learning",
            "message": f"Actively learning from {total_samples} comparisons",
            "is_learning": True,
            "total_samples": total_samples,
            "mae": mae,
            "rmse": rmse,
            "mean_bias": mean_bias,
            "mean_pct_error": mean_pct_error,
            "learning_confidence": learning_confidence,
            "slots_tracked": len(self.error_metrics),
            "bias_corrections": len(self.bias_correction),
            "hourly_metrics": slot_metrics,
            "pending_predictions": pending,
        }

    def get_hourly_error_report(self) -> list[dict]:
        """Get detailed error report for each 15-minute slot.

        Returns:
            List of dictionaries with per-slot error statistics
        """
        report = []

        for slot in range(96):
            metrics = self.error_metrics.get(slot)
            hour = slot // 4
            minute = (slot % 4) * 15

            if metrics and metrics["count"] > 0:
                report.append(
                    {
                        "slot": slot,
                        "hour": hour,
                        "minute": minute,
                        "samples": metrics["count"],
                        "mae": float(np.mean(metrics["abs_errors"])),
                        "mean_error": float(np.mean(metrics["errors"])),
                        "std_error": float(np.std(metrics["errors"])),
                        "bias_correction": self.bias_correction.get(slot, 0.0),
                        "overpredicts": sum(1 for e in metrics["errors"] if e > 0),
                        "underpredicts": sum(1 for e in metrics["errors"] if e < 0),
                    }
                )
            else:
                report.append(
                    {
                        "slot": slot,
                        "hour": hour,
                        "minute": minute,
                        "samples": 0,
                        "message": "No data yet",
                    }
                )

        return report

    async def reset_learning(self) -> None:
        """Reset all learning data and start fresh.

        Use this if you want to clear accumulated learning and start over.
        """
        self.error_metrics = {}
        self.bias_correction = {}
        self.lead_time_accuracy = {}

        # Clear storage file
        await self.storage.async_clear_storage()

        _LOGGER.info("Self-learning data has been reset")

    async def save_learning_data(self) -> bool:
        """Save current learning data to persistent storage."""
        data = {
            "bias_correction": self.bias_correction,
            "error_metrics": self.error_metrics,
            "volatility_mae": getattr(self, "volatility_mae", {}),
            "price_history": self.price_history,
            "training_samples": self.training_samples,
            "is_trained": self.is_trained,
            "solar_scale": getattr(self, "solar_scale", 1.0),
            "solar_scale_samples": getattr(self, "_solar_scale_samples", 0),
        }

        return await self.storage.async_save_all(data)

    async def _load_learning_data(self) -> None:
        """Load learning data from persistent storage.

        Called automatically during initialization. Also restores
        optimized hyperparameters if previously saved.
        """
        data = await self.storage.async_load_all()

        if data:
            self.bias_correction = data.get("bias_correction", {})
            self.error_metrics = data.get("error_metrics", {})
            self.volatility_mae = data.get("volatility_mae", {})
            self.price_history = data.get("price_history", [])
            self.training_samples = data.get("training_samples", 0)
            self.is_trained = data.get("is_trained", False)
            self.solar_scale = data.get("solar_scale", 1.0)
            self._solar_scale_samples = data.get("solar_scale_samples", 0)
            self._restore_hpo_counter(data)
            self._restore_holdout_metrics(data)

            # Restore optimized hyperparameters if available. Results saved
            # without hpo_max_depth were tuned for the old depth-1 stumps and
            # are ignored until the next optimization run.
            hpo_n = data.get("hpo_n_estimators")
            hpo_lr = data.get("hpo_learning_rate")
            hpo_depth = data.get("hpo_max_depth")
            if hpo_n and hpo_lr and hpo_depth:
                try:
                    from .models import create_price_model

                    self.price_model = create_price_model(
                        n_estimators=int(hpo_n),
                        learning_rate=float(hpo_lr),
                        max_depth=int(hpo_depth),
                    )
                    self.is_trained = False  # Force re-train with new params
                    _LOGGER.info(
                        "Restored optimized hyperparameters: n=%d, lr=%.2f, depth=%d",
                        int(hpo_n),
                        float(hpo_lr),
                        int(hpo_depth),
                    )
                except ValueError, TypeError:
                    pass
            elif hpo_n and hpo_lr:
                _LOGGER.debug(
                    "Ignoring hyperparameters tuned for the old stump model; "
                    "using defaults until the next optimization"
                )

            pred_count = data.get("prediction_count", 0)

            _LOGGER.info(
                "Restored learning state: %d bias corrections, %d error metrics, "
                "%d volatility slots, %d pending predictions, %d days price history",
                len(self.bias_correction),
                len(self.error_metrics),
                len(getattr(self, "volatility_mae", {})),
                pred_count,
                len(self.price_history),
            )

            # Catch-up replay: if learned samples are tiny compared to pending
            # predictions (e.g. after schema migration wiped error_metrics), replay
            # all stored predictions against known price history in one pass.
            pending = self.storage.count_predictions()
            total_learned = sum(m.get("count", 0) for m in self.error_metrics.values())
            needs_catchup = (
                pending > 100 and total_learned < 100 and len(self.price_history) >= 1
            )
            if needs_catchup:
                _LOGGER.info(
                    "Catch-up needed: %d learned vs %d pending predictions "
                    "with %d days price history — starting replay",
                    total_learned,
                    pending,
                    len(self.price_history),
                )
                learned = await self.hass.async_add_executor_job(self.catch_up_learning)
                if learned > 0:
                    _LOGGER.info(
                        "Catch-up replay complete: %d predictions learned, saving",
                        learned,
                    )
                    await self.save_learning_data()
