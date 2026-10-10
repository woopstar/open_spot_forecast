"""Record external price forecasts and score them like the model's own (#120).

Other integrations publish forecasts too (Stromligning's forecast sensor,
Energi Data Service's Carnot forecast). They cannot be backtested, so the
only way to compare them with the model is to record them live: every
forecast run stores what each configured source shows for the slots the
model predicts, and when self-learning scores a slot it scores the sources'
stored forecasts for it with the same lead-time buckets. The model's own
metrics, bias correction and predictions never read any of it.

A blend of the model with a source can only be scored from both errors of the
same slot, which the daily sums do not hold. So a scored slot's mean error per
bucket is kept for the model and for every source, for the same rolling
window (#157); ``scripts/live_report.py`` evaluates the blends from an export.

The rows are raw spot prices in currency/kWh, the unit of the model's
errors: the component layer converts the displayed prices before storing
them (``external_forecasts.py`` next to the updater).
"""

import logging
import sqlite3
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from typing import Any

from homeassistant.util import dt as dt_util

from ..const import EXTERNAL_MODEL_SOURCE, LEAD_TIME_WINDOW_DAYS
from ..time_slots import utc_slot_key
from .base import PredictorBase
from .lead_time import bucket_errors, summarize_error_sums

_LOGGER = logging.getLogger(__name__)


def _mean_errors(
    source: str, errors_by_bucket: Mapping[str, Sequence[float]]
) -> list[tuple[str, str, int, float]]:
    """Return a source's ``external_slot_errors`` rows for one slot.

    Args:
        source: The source's name.
        errors_by_bucket: Signed errors (forecast - actual) per bucket key.

    Returns:
        ``(bucket, source, number of forecasts, mean signed error)`` for every
        bucket with a forecast.
    """
    return [
        (bucket, source, len(errors), sum(errors) / len(errors))
        for bucket, errors in errors_by_bucket.items()
        if errors
    ]


class ExternalForecastMixin(PredictorBase):
    """Store external forecasts and keep their accuracy per lead time."""

    def store_external_forecasts(
        self, forecasts: Mapping[str, Sequence[tuple[str, float]]]
    ) -> None:
        """Store the sources' forecasts as read now (blocking).

        Readings older than the training window (a source that stopped, or
        slots that were never scored) are pruned, like stored predictions. A
        storage failure is logged and never interrupts the forecast.

        Args:
            forecasts: Per source, ``(slot's UTC key, raw spot price)`` rows.
        """
        now = dt_util.utcnow()
        try:
            for source, rows in forecasts.items():
                # The exact time, like a prediction's: a lead time rounded to
                # the slot would fall into the next bucket at 24/48/72 hours
                self.storage.insert_external_forecasts(source, now.isoformat(), rows)
            self.storage.delete_external_forecasts_stored_before(
                utc_slot_key(now - timedelta(days=self.max_history_days))
            )
        except sqlite3.Error as err:
            _LOGGER.error("Failed to store the external forecasts: %s", err)

    def record_external_accuracy(
        self,
        slot_start: datetime,
        predictions: list[dict[str, Any]],
        actual_price: float,
    ) -> None:
        """Score the stored external forecasts of a slot and remove them (blocking).

        Each forecast's signed error goes into its source's lead-time bucket
        (``bucket_errors``, as the model's own). The slot's mean error per
        bucket is also kept for every source, and for the model in the
        buckets a source has (#157). A storage failure is logged and never
        interrupts the learning loop.

        Args:
            slot_start: Start of the scored slot; naive is Home Assistant's
                local time.
            predictions: The model's own matched predictions for the slot,
                only read to keep their errors next to the sources'.
            actual_price: Actual raw spot price of the slot.
        """
        key = utc_slot_key(dt_util.as_utc(slot_start))
        try:
            by_source: dict[str, list[dict[str, Any]]] = {}
            for row in self.storage.find_external_forecasts(key):
                by_source.setdefault(row["source"], []).append(row)
            if not by_source:
                return
            # The slot's local date, as lead_time_accuracy keys its rows
            slot_date = dt_util.as_local(slot_start).date().isoformat()
            slot_errors: list[tuple[str, str, int, float]] = []
            for source, forecasts in by_source.items():
                errors = bucket_errors(forecasts, actual_price)
                self.storage.add_external_errors(slot_date, source, errors)
                slot_errors += _mean_errors(source, errors)
            paired = {bucket for bucket, *_ in slot_errors}
            own = bucket_errors(predictions, actual_price)
            slot_errors += _mean_errors(
                EXTERNAL_MODEL_SOURCE,
                {bucket: errors for bucket, errors in own.items() if bucket in paired},
            )
            self.storage.upsert_external_slot_errors(key, slot_errors)
            self.storage.delete_external_forecasts(key)
            self.refresh_external_accuracy()
        except sqlite3.Error as err:
            _LOGGER.error("Failed to record the external forecast accuracy: %s", err)

    def refresh_external_accuracy(self) -> None:
        """Prune rows outside the rolling window and reload the summary (blocking)."""
        today = dt_util.now().date()
        cutoff = (today - timedelta(days=LEAD_TIME_WINDOW_DAYS - 1)).isoformat()
        self.storage.delete_external_accuracy_before(cutoff)
        self.storage.delete_external_slot_errors_before(
            utc_slot_key(dt_util.utcnow() - timedelta(days=LEAD_TIME_WINDOW_DAYS))
        )
        self.external_accuracy = {
            source: summarize_error_sums(sums)
            for source, sums in self.storage.get_external_error_sums(cutoff).items()
        }
