"""Bias correction: an additive offset per 15-minute slot and lead-time bucket.

A slot's offset in a bucket is the mean of the raw model's errors in that
bucket over the slot's last ``BIAS_WINDOW_DAYS`` measured days (#203):

    raw_error            = stored price + stored bias_offset - actual
    offset[slot][bucket] = mean(raw errors of the bucket's window)
    corrected            = raw_prediction - offset[slot][bucket]

Every stored prediction keeps the offset that was subtracted from it
(``predictions.bias_offset``), so the raw model's error can be measured
whatever correction was in use. The offsets before #203 were an exponential
moving average of the errors left *after* the correction; they followed the
bias with a lag of about ten days while the bias changes sign within a few,
and then added to the error they were meant to remove.

An error can only be measured when its slot arrives, a lead time after the
forecast it corrects next: a day for ``day_1``, two for ``day_2``, three to
eight for the later buckets. The lead-time bias does not last that long
(``docs/self_learning.md`` "Bias Correction"), so only the buckets in
``BIAS_WINDOW_DAYS`` have an offset. A window without errors from its last
days gives none, and no bucket borrows another's.

No bucket is listed since #206: on 14 live DK1 days the 4-day offsets of
``day_1`` and ``day_2`` raised their error, and so did every other window,
pooled or per slot. The machinery stays, so a bucket can be corrected again
by listing it.
"""

from collections.abc import Mapping, Sequence
from datetime import date, timedelta
from typing import Any

from ..const import BIAS_FALLBACK_BUCKET
from .base import PredictorBase
from .lead_time import raw_bucket_errors

# Days of raw errors a bucket's offset is the mean of, per slot. A bucket
# that is not listed is not corrected. Empty since #206: the raw model's
# lead-time bias does not last until its errors are measured, and every
# trailing offset on the live DK1 export added error (#203 had day_1/day_2: 4)
BIAS_WINDOW_DAYS: dict[str, int] = {}
# Fewer raw errors in the window than this give no offset
MIN_BIAS_SAMPLES = 3
# A slot's metrics key: {bucket: {ISO day: [samples, sum of raw errors]}}
RAW_ERRORS_KEY = "raw_errors"


def add_raw_errors(
    metrics: dict[str, Any],
    predictions: list[dict[str, Any]],
    actual_price: float,
    day: date,
) -> None:
    """Add the matched predictions' raw errors to a slot's bias windows.

    Kept as a count and a sum per bucket and measured day, for the buckets
    in ``BIAS_WINDOW_DAYS`` and their window's number of days. A prediction
    stored before #203 has no ``bias_offset``: its raw error is unknown and
    it is left out.

    Args:
        metrics: The slot's error metrics (see ``new_slot_metrics``).
        predictions: Stored prediction rows matched for the slot.
        actual_price: The slot's actual price.
        day: The slot's local date.
    """
    windows = metrics.setdefault(RAW_ERRORS_KEY, {})
    for bucket, pairs in raw_bucket_errors(predictions, actual_price).items():
        if bucket not in BIAS_WINDOW_DAYS:
            continue
        days = windows.setdefault(bucket, {})
        samples, total = days.get(day.isoformat(), (0, 0.0))
        days[day.isoformat()] = [
            samples + len(pairs),
            total + sum(raw for raw, _corrected in pairs),
        ]
    for bucket in list(windows):
        kept = sorted(windows[bucket])[-BIAS_WINDOW_DAYS.get(bucket, 0) :]
        windows[bucket] = {key: windows[bucket][key] for key in kept}
        if bucket not in BIAS_WINDOW_DAYS:
            del windows[bucket]


def window_offset(
    days: Mapping[str, Sequence[float]], today: date, window_days: int
) -> float | None:
    """Return the mean raw error of a bucket's window as of ``today``.

    Args:
        days: ``{ISO day: (samples, sum of raw errors)}`` of one slot and bucket.
        today: The local date the offset is used on.
        window_days: The window: the last this many measured days, none of
            them more than this many days before ``today``.

    Returns:
        The offset, or None with fewer than ``MIN_BIAS_SAMPLES`` errors in
        the window: a bucket without recent errors is not corrected.
    """
    oldest = (today - timedelta(days=window_days)).isoformat()
    recent = [days[key] for key in sorted(days) if oldest <= key <= today.isoformat()][
        -window_days:
    ]
    samples = sum(int(day[0]) for day in recent)
    if samples < MIN_BIAS_SAMPLES:
        return None
    return sum(float(day[1]) for day in recent) / samples


def slot_offsets(metrics: Mapping[str, Any], today: date) -> dict[str, float]:
    """Return a slot's offset per lead-time bucket as of ``today``.

    Args:
        metrics: The slot's error metrics.
        today: The local date the offsets are used on.

    Returns:
        ``{bucket: offset}`` for the buckets whose window has recent errors.
    """
    windows = metrics.get(RAW_ERRORS_KEY) or {}
    offsets: dict[str, float] = {}
    for bucket, window_days in BIAS_WINDOW_DAYS.items():
        offset = window_offset(windows.get(bucket) or {}, today, window_days)
        if offset is not None:
            offsets[bucket] = offset
    return offsets


class BiasMixin(PredictorBase):
    """The learned bias offsets of ``SpotPricePredictor``."""

    def _bias_offset(self, slot: int, bucket: str) -> float | None:
        """Return the offset of a slot and lead-time bucket, or None without one.

        Args:
            slot: 15-minute slot index (0-95)
            bucket: ``LEAD_TIME_BUCKETS`` key

        Returns:
            The offset to subtract from a raw prediction. A bucket without
            one is not corrected; it never uses another bucket's offset.
        """
        return (self.bias_correction.get(slot) or {}).get(bucket)

    def apply_bias_correction(
        self, predicted_price: float, slot: int, bucket: str = BIAS_FALLBACK_BUCKET
    ) -> float:
        """Subtract the bias offset of a slot and lead time from a prediction.

        The correction is additive, so it works the same for positive, zero
        and negative prices and never flips a prediction's sign by scaling
        it. The result is not clamped: prices can be negative.

        Args:
            predicted_price: Raw model prediction
            slot: 15-minute slot index (0-95)
            bucket: Lead-time bucket of the prediction

        Returns:
            ``predicted_price - offset[slot][bucket]`` (unchanged without an
            offset)
        """
        return predicted_price - (self._bias_offset(slot, bucket) or 0.0)

    def _update_bias_correction(self, slot: int, today: date) -> None:
        """Set a slot's offsets from its windows of raw errors as of ``today``.

        Runs when the slot is matched and, for every slot, before a forecast
        run (``refresh_bias_offsets``). An offset whose window has no recent
        errors is dropped, not kept from an earlier day.

        Args:
            slot: 15-minute slot index (0-95)
            today: The local date the offsets are used on.
        """
        offsets = slot_offsets(self.error_metrics.get(slot) or {}, today)
        if offsets:
            self.bias_correction[slot] = offsets
        else:
            self.bias_correction.pop(slot, None)

    def refresh_bias_offsets(self, today: date) -> None:
        """Bring every slot's offsets up to ``today`` before they are used.

        After days without a match (the instance was off, the price source
        was down) the stored offsets describe an older bias: they are
        recomputed, and dropped where the window is stale.
        """
        for slot in set(self.error_metrics) | set(self.bias_correction):
            self._update_bias_correction(slot, today)
