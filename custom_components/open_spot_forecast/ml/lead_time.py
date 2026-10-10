"""Live forecast accuracy per lead time (day 1/2/3/4+).

Every stored prediction records when it was made (``stored_at``) and the
15-minute slot it targets (``start``). When the actual price of a slot is
known, the signed error of each matched prediction is bucketed by its lead
time (``start - stored_at``) and added to daily per-bucket sums in SQLite.
MAE, RMSE and bias per bucket are then reported over a rolling window.

The prediction made closest to ``EVALUATION_LEAD_HOURS`` before its slot is
also kept next to the actual price (``evaluation`` table, #36), so the
predicted and the actual series can be charted side by side. So are the
snapshots at the other ``EVALUATION_LEAD_TIMES`` (#113), when a prediction
was made close enough to them. Before a slot is scored, the same choice among
its stored predictions is the slot's day-ahead prediction
(``day_ahead_prediction()``), the state of the day-ahead prediction sensor.
"""

import logging
import math
import sqlite3
from datetime import UTC, datetime, timedelta
from typing import Any

from homeassistant.util import dt as dt_util

from ..const import (
    BIAS_FALLBACK_BUCKET,
    DAY_AHEAD_PREDICTION_HOURS,
    EVALUATION_KEEP_DAYS,
    EVALUATION_LEAD_HOURS,
    EVALUATION_LEAD_TIMES,
    LEAD_TIME_BUCKETS,
    LEAD_TIME_WINDOW_DAYS,
)
from ..time_slots import (
    SLOT_MINUTES,
    UTC_KEY_FORMAT,
    floor_to_slot,
    parse_utc,
    utc_slot_key,
)
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


def snapshot_tolerance(target_hours: float) -> float | None:
    """Return how far from a lead time a prediction may be to be its snapshot.

    Args:
        target_hours: One of ``EVALUATION_LEAD_TIMES``.

    Returns:
        Half the gap to the nearest other lead time; None for
        ``EVALUATION_LEAD_HOURS``, whose snapshot is always kept (#36).
    """
    if abs(target_hours - EVALUATION_LEAD_HOURS) < 1e-9:
        return None
    gaps = [
        abs(other - target_hours)
        for other in EVALUATION_LEAD_TIMES
        if abs(other - target_hours) > 1e-9
    ]
    return min(gaps) / 2 if gaps else None


def evaluation_snapshots(
    predictions: list[dict[str, Any]],
) -> dict[float, tuple[dict[str, Any], float]]:
    """Return a slot's prediction per lead time in ``EVALUATION_LEAD_TIMES`` (#113).

    A lead time without a prediction made within its ``snapshot_tolerance()``
    is left out, so a 12 h snapshot is never a prediction made 31 h ahead.

    Args:
        predictions: Matched prediction rows for one slot (see
            ``evaluation_prediction``).

    Returns:
        ``{target_hours: (prediction, lead_hours)}`` for the lead times that
        have a snapshot.
    """
    snapshots: dict[float, tuple[dict[str, Any], float]] = {}
    for target in EVALUATION_LEAD_TIMES:
        chosen = evaluation_prediction(predictions, target)
        if chosen is None:
            continue
        tolerance = snapshot_tolerance(target)
        if tolerance is None or abs(chosen[1] - target) <= tolerance:
            snapshots[target] = chosen
    return snapshots


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

        The snapshots at the other ``EVALUATION_LEAD_TIMES`` are kept too
        (#113). A storage failure is logged and never interrupts the
        learning loop.

        Args:
            slot_start: Start of the matched slot; naive is Home Assistant's
                local time.
            predictions: All stored predictions matched for the slot.
            actual_price: Actual price of the slot.
        """
        start = dt_util.as_utc(slot_start)
        snapshots = evaluation_snapshots(predictions)
        # Older slots (the startup catch-up) would be pruned right away
        if not snapshots or start < dt_util.utcnow() - timedelta(
            days=EVALUATION_KEEP_DAYS
        ):
            return
        try:
            for target, (prediction, lead) in snapshots.items():
                self.storage.upsert_evaluation(
                    utc_slot_key(start),
                    float(prediction["price"]),
                    actual_price,
                    round(lead, 2),
                    target,
                )
            self.refresh_evaluation()
        except sqlite3.Error as err:
            _LOGGER.error("Failed to record the evaluation: %s", err)

    def refresh_evaluation(self) -> None:
        """Prune slots older than the kept days and reload the series (blocking).

        ``evaluation`` is the ``EVALUATION_LEAD_HOURS`` series;
        ``evaluation_snapshots`` holds every lead time's (#113).
        """
        cutoff = utc_slot_key(dt_util.utcnow() - timedelta(days=EVALUATION_KEEP_DAYS))
        self.storage.delete_evaluation_before(cutoff)
        snapshots: dict[float, list[dict[str, Any]]] = {}
        for target in EVALUATION_LEAD_TIMES:
            series = snapshots[target] = []
            for row in self.storage.get_evaluation(cutoff, target):
                start = datetime.strptime(row["timestamp"], UTC_KEY_FORMAT).replace(
                    tzinfo=UTC
                )
                end = start + timedelta(minutes=SLOT_MINUTES)
                series.append(
                    {
                        "start": dt_util.as_local(start).isoformat(),
                        "end": dt_util.as_local(end).isoformat(),
                        "predicted": row["predicted"],
                        "actual": row["actual"],
                        "lead_hours": row["lead_hours"],
                    }
                )
        self.evaluation_snapshots = snapshots
        self.evaluation = snapshots[EVALUATION_LEAD_HOURS]

    def refresh_day_ahead_predictions(self) -> None:
        """Reload the day-ahead prediction of the current and coming slots (blocking).

        For every slot of the next ``DAY_AHEAD_PREDICTION_HOURS`` that has
        stored predictions, the one ``evaluation_prediction()`` would keep
        once the slot is scored. Run after every forecast (the only writer of
        predictions) and at startup. A storage failure is logged and keeps
        the cached ones.
        """
        first = dt_util.utcnow()
        try:
            rows = self.storage.get_predictions_between(
                utc_slot_key(first),
                utc_slot_key(first + timedelta(hours=DAY_AHEAD_PREDICTION_HOURS)),
            )
        except sqlite3.Error as err:
            _LOGGER.error("Failed to load the day-ahead predictions: %s", err)
            return
        by_slot: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            start = parse_utc(row["start"])
            if start is not None:
                by_slot.setdefault(utc_slot_key(start), []).append(row)
        self.day_ahead_predictions = {
            key: float(chosen[0]["price"])
            for key, predictions in by_slot.items()
            if (chosen := evaluation_prediction(predictions)) is not None
        }

    def day_ahead_prediction(self, slot_start: datetime) -> float | None:
        """Return the prediction made about a day before a slot (raw spot price).

        Args:
            slot_start: Timezone-aware start of a current or recent slot.

        Returns:
            The slot's cached day-ahead prediction; once the slot is scored
            (its stored predictions are gone), the one kept in the
            evaluation; None if the slot was never predicted.
        """
        start = floor_to_slot(dt_util.as_utc(slot_start))
        if (price := self.day_ahead_predictions.get(utc_slot_key(start))) is not None:
            return price
        # The newest slots are last
        for row in reversed(self.evaluation):
            row_start = parse_utc(row["start"])
            if row_start is None or row_start < start:
                break
            if row_start == start:
                return float(row["predicted"])
        return None

    def refresh_lead_time_accuracy(self) -> None:
        """Prune rows outside the rolling window and reload the summary (blocking)."""
        today = dt_util.now().date()
        cutoff = (today - timedelta(days=LEAD_TIME_WINDOW_DAYS - 1)).isoformat()
        self.storage.delete_lead_time_accuracy_before(cutoff)
        self.lead_time_accuracy = summarize_error_sums(
            self.storage.get_lead_time_error_sums(cutoff)
        )
