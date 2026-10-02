"""Live forecast accuracy per lead time (day 1/2/3/4+).

Every stored prediction records when it was made (``stored_at``) and the
15-minute slot it targets (``start``). When the actual price of a slot is
known, the signed error of each matched prediction is bucketed by its lead
time (``start - stored_at``) and added to daily per-bucket sums in SQLite.
MAE, RMSE and bias per bucket are then reported over a rolling window.

The prediction made closest to ``EVALUATION_LEAD_HOURS`` before its slot is
also kept next to the actual price (``evaluation`` table, #36), so the
predicted and the actual series can be charted side by side.
"""

import logging
import math
import sqlite3
from datetime import UTC, datetime, timedelta
from typing import Any

from homeassistant.util import dt as dt_util

from ..const import (
    BIAS_FALLBACK_BUCKET,
    EVALUATION_KEEP_DAYS,
    EVALUATION_LEAD_HOURS,
    LEAD_TIME_BUCKETS,
    LEAD_TIME_WINDOW_DAYS,
)
from ..time_slots import SLOT_MINUTES, UTC_KEY_FORMAT, utc_slot_key
from .base import PredictorBase

_LOGGER = logging.getLogger(__name__)


def lead_time_hours(slot_start: str, stored_at: str) -> float | None:
    """Return the hours between storing a prediction and the slot it targets.

    Naive timestamps (``stored_at`` from before it was stored with its UTC
    offset) are interpreted in Home Assistant's local time zone.

    Args:
        slot_start: ISO start of the predicted slot.
        stored_at: ISO time the prediction was stored.

    Returns:
        Lead time in hours, or None if either timestamp cannot be parsed.
    """
    start = dt_util.parse_datetime(slot_start)
    stored = dt_util.parse_datetime(stored_at)
    if start is None or stored is None:
        return None
    return (dt_util.as_utc(start) - dt_util.as_utc(stored)).total_seconds() / 3600


def lead_time_bucket(lead_hours: float) -> str | None:
    """Return the ``LEAD_TIME_BUCKETS`` key a lead time falls into.

    Args:
        lead_hours: Lead time in hours.

    Returns:
        Bucket key, or None for a negative lead time (a prediction stored after
        its slot started is not a forecast).
    """
    if lead_hours < 0:
        return None
    for bucket, upper_hours in LEAD_TIME_BUCKETS:
        if lead_hours < upper_hours:
            return bucket
    return None


def prediction_bucket(slot_start: str, now: datetime) -> str:
    """Return the lead-time bucket of a prediction made ``now`` for a slot.

    Used to pick the bias offset of the lead time at prediction time (#118),
    the bucket the prediction's error is later learned in.

    Args:
        slot_start: ISO start of the predicted slot.
        now: When the prediction is made (``stored_at``).

    Returns:
        The bucket key; ``BIAS_FALLBACK_BUCKET`` for the slot already under
        way (negative lead time) or an unreadable start.
    """
    lead = lead_time_hours(slot_start, now.isoformat())
    bucket = lead_time_bucket(lead) if lead is not None else None
    return bucket or BIAS_FALLBACK_BUCKET


def bucket_errors(
    predictions: list[dict[str, Any]], actual_price: float
) -> dict[str, list[float]]:
    """Group signed errors (predicted - actual) by lead-time bucket.

    Args:
        predictions: Matched prediction rows with ``start``, ``stored_at`` and
            ``price``.
        actual_price: Actual price of the slot.

    Returns:
        Signed errors per bucket key. Predictions without a valid lead time
        are skipped.
    """
    errors: dict[str, list[float]] = {}
    for prediction in predictions:
        lead = lead_time_hours(
            str(prediction.get("start", "")), str(prediction.get("stored_at", ""))
        )
        bucket = lead_time_bucket(lead) if lead is not None else None
        if bucket is not None:
            error = float(prediction.get("price", 0)) - actual_price
            errors.setdefault(bucket, []).append(error)
    return errors


def evaluation_prediction(
    predictions: list[dict[str, Any]], target_hours: float = EVALUATION_LEAD_HOURS
) -> tuple[dict[str, Any], float] | None:
    """Return the prediction made closest to ``target_hours`` before its slot.

    Args:
        predictions: Matched prediction rows for one slot, with ``start``,
            ``stored_at`` and ``price``.
        target_hours: The lead time to evaluate (a day ahead by default).

    Returns:
        The prediction and its lead time in hours; None if no prediction has
        a valid, non-negative lead time. Ties go to the later prediction.
    """
    best: tuple[dict[str, Any], float] | None = None
    for prediction in predictions:
        lead = lead_time_hours(
            str(prediction.get("start", "")), str(prediction.get("stored_at", ""))
        )
        if lead is None or lead < 0:
            continue
        if best is None or abs(lead - target_hours) <= abs(best[1] - target_hours):
            best = (prediction, lead)
    return best


def summarize_error_sums(
    sums: dict[str, tuple[int, float, float, float]],
) -> dict[str, dict[str, float | int]]:
    """Turn per-bucket error sums into MAE, RMSE, bias and sample count.

    Args:
        sums: ``{bucket: (samples, sum_error, sum_abs_error, sum_sq_error)}``.

    Returns:
        ``{bucket: {"mae", "rmse", "bias", "samples"}}`` for buckets with samples.
    """
    summary: dict[str, dict[str, float | int]] = {}
    for bucket, (samples, sum_error, sum_abs_error, sum_sq_error) in sums.items():
        if samples > 0:
            summary[bucket] = {
                "mae": sum_abs_error / samples,
                "rmse": math.sqrt(sum_sq_error / samples),
                "bias": sum_error / samples,
                "samples": samples,
            }
    return summary


class LeadTimeMixin(PredictorBase):
    """Record and summarize live forecast accuracy per lead time."""

    def record_lead_time_accuracy(
        self, predictions: list[dict[str, Any]], actual_price: float
    ) -> None:
        """Persist matched prediction errors per lead-time bucket (blocking).

        Also refreshes the cached summary read by the accuracy sensors. A
        storage failure is logged and never interrupts the learning loop.

        Args:
            predictions: All stored predictions matched for one slot.
            actual_price: Actual price of that slot.
        """
        try:
            errors = bucket_errors(predictions, actual_price)
            if errors:
                # All matched predictions target the same slot, hence one date.
                slot_date = str(predictions[0]["start"])[:10]
                self.storage.add_lead_time_errors(slot_date, errors)
            self.refresh_lead_time_accuracy()
        except sqlite3.Error as err:
            _LOGGER.error("Failed to record lead-time accuracy: %s", err)

    def record_evaluation(
        self,
        slot_start: datetime,
        predictions: list[dict[str, Any]],
        actual_price: float,
    ) -> None:
        """Keep the slot's day-ahead prediction next to its actual price (blocking).

        A storage failure is logged and never interrupts the learning loop.

        Args:
            slot_start: Start of the matched slot; naive is Home Assistant's
                local time.
            predictions: All stored predictions matched for the slot.
            actual_price: Actual price of the slot.
        """
        start = dt_util.as_utc(slot_start)
        chosen = evaluation_prediction(predictions)
        # Older slots (the startup catch-up) would be pruned right away
        if chosen is None or start < dt_util.utcnow() - timedelta(
            days=EVALUATION_KEEP_DAYS
        ):
            return
        prediction, lead = chosen
        try:
            self.storage.upsert_evaluation(
                utc_slot_key(start),
                float(prediction["price"]),
                actual_price,
                round(lead, 2),
            )
            self.refresh_evaluation()
        except sqlite3.Error as err:
            _LOGGER.error("Failed to record the evaluation: %s", err)

    def refresh_evaluation(self) -> None:
        """Prune slots older than the kept days and reload the series (blocking)."""
        cutoff = utc_slot_key(dt_util.utcnow() - timedelta(days=EVALUATION_KEEP_DAYS))
        self.storage.delete_evaluation_before(cutoff)
        evaluation = []
        for row in self.storage.get_evaluation(cutoff):
            start = datetime.strptime(row["timestamp"], UTC_KEY_FORMAT).replace(
                tzinfo=UTC
            )
            end = start + timedelta(minutes=SLOT_MINUTES)
            evaluation.append(
                {
                    "start": dt_util.as_local(start).isoformat(),
                    "end": dt_util.as_local(end).isoformat(),
                    "predicted": row["predicted"],
                    "actual": row["actual"],
                    "lead_hours": row["lead_hours"],
                }
            )
        self.evaluation = evaluation

    def refresh_lead_time_accuracy(self) -> None:
        """Prune rows outside the rolling window and reload the summary (blocking)."""
        today = dt_util.now().date()
        cutoff = (today - timedelta(days=LEAD_TIME_WINDOW_DAYS - 1)).isoformat()
        self.storage.delete_lead_time_accuracy_before(cutoff)
        self.lead_time_accuracy = summarize_error_sums(
            self.storage.get_lead_time_error_sums(cutoff)
        )
