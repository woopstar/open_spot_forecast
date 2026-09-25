"""Catch-up learning: replay stored predictions against known prices at startup."""

import logging
from datetime import datetime

import numpy as np

from ..time_slots import slot_start_in_day
from .base import PredictorBase
from .learning import percent_error, predictions_at_instant

_LOGGER = logging.getLogger(__name__)


class CatchUpMixin(PredictorBase):
    """Rebuild self-learning state from stored predictions in one pass.

    Designed to be mixed into SpotPricePredictor — all attributes
    referenced via self are provided by the owning class.
    """

    def catch_up_learning(self) -> int:
        """Replay all stored predictions against known historical prices.

        Called at startup when error_metrics is empty but the predictions
        table has data (e.g. after schema migration cleared learned data).
        Iterates over every day in price_history, matches stored predictions,
        and rebuilds error_metrics and bias_correction in one pass.

        Returns:
            Number of predictions matched and learned from.
        """
        total_learned = 0

        # Get all pending prediction dates from the DB
        pending_dates = self.storage.get_pending_prediction_dates()
        if not pending_dates:
            return 0

        _LOGGER.info(
            "Starting catch-up learning across %d dates with %d days of price history",
            len(pending_dates),
            len(self.price_history),
        )

        # Build price lookup by date
        price_by_date: dict[str, list[float | None]] = {}
        for entry in self.price_history:
            date_str = entry.get("date")
            prices = entry.get("prices", [])
            if date_str and prices:
                price_by_date[date_str] = prices

        for date_str in sorted(pending_dates):
            prices = price_by_date.get(date_str)
            if not prices:
                continue

            day = datetime.strptime(date_str, "%Y-%m-%d").date()
            for idx, actual_price in enumerate(prices):
                # A slot missing in the source (None) has nothing to learn from
                if actual_price is None or actual_price == 0:
                    continue

                # Real slot start: correct on DST days, and indexes past the
                # day's end (tomorrow's prices stored with today's) land on
                # the next date
                slot_start = slot_start_in_day(day, idx, self.tz)
                hour = slot_start.hour
                minute = slot_start.minute

                # Find matching predictions for this slot
                matching = predictions_at_instant(
                    self.storage.find_predictions_for_timestamp(
                        slot_start.strftime("%Y-%m-%d"), hour, minute
                    ),
                    slot_start,
                )
                if not matching:
                    continue

                # Initialize error tracking slot
                slot = hour * 4 + minute // 15
                if slot not in self.error_metrics:
                    self.error_metrics[slot] = {
                        "errors": [],
                        "abs_errors": [],
                        "pct_errors": [],
                        "predictions": [],
                        "actuals": [],
                        "count": 0,
                    }
                metrics = self.error_metrics[slot]

                for prediction in matching:
                    predicted_price = prediction.get("price", 0)
                    error = predicted_price - actual_price
                    abs_error = abs(error)
                    pct_error = percent_error(error, actual_price)

                    metrics["errors"].append(error)
                    metrics["abs_errors"].append(abs_error)
                    metrics["pct_errors"].append(pct_error)
                    metrics["predictions"].append(predicted_price)
                    metrics["actuals"].append(actual_price)
                    metrics["count"] += 1

                    total_learned += 1

                self.record_lead_time_accuracy(matching, actual_price)

                # Update bias correction after processing this slot
                self._update_bias_correction(slot)

                # Initialize volatility for this slot
                if matching and slot not in self.volatility_mae:
                    abs_errors = self.error_metrics[slot]["abs_errors"]
                    self.volatility_mae[slot] = float(np.mean(abs_errors))

                # Remove matched predictions
                for p in matching:
                    self.storage.remove_prediction(p["id"])

                # Keep error arrays bounded
                max_samples = 100
                for key in [
                    "errors",
                    "abs_errors",
                    "pct_errors",
                    "predictions",
                    "actuals",
                ]:
                    if len(metrics[key]) > max_samples:
                        metrics[key] = metrics[key][-max_samples:]

        _LOGGER.info(
            "Catch-up learning complete: %d predictions matched across %d slots",
            total_learned,
            len(self.error_metrics),
        )
        return total_learned
