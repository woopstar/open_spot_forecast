"""What a blend of the ML forecast with the external forecasts would have scored (#157).

An ensemble of independent forecasts often beats each member. Whether a
weighted mean of the ML forecast and the recorded external forecasts
(Stromligning, Energi Data Service; #120) does is measured here, before any
blend is built: from the per-slot errors a running instance keeps in
``external_slot_errors``, read by ``scripts/live_report.py`` from an export.

Per lead-time bucket and over the slots the model and every source of the
bucket have, it scores the model, each source, and two blends:

- **equal weights**: the mean of the model and the sources;
- **inverse MSE**: each member weighted by 1 / its mean squared error. The
  weights of a day come from the earlier days only, and the blend is the
  model alone until ``MIN_BLEND_SLOTS`` earlier slots exist, as a shipped
  blend would have to work. Weights fitted on the slots they are scored on
  would flatter the blend.

A member's error in a slot is the mean signed error of its forecasts in the
bucket, so a blend's error is the weighted sum of its members' errors. Dev-only,
not shipped with the integration.
"""

import math
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date

from custom_components.open_spot_forecast.const import EXTERNAL_MODEL_SOURCE

# (slot's UTC key, bucket) -> source -> mean signed error (forecast - actual)
SlotErrors = dict[tuple[str, str], dict[str, float]]

# Paired slots of earlier days the inverse-MSE weights need: three days
MIN_BLEND_SLOTS = 3 * 96
EQUAL_BLEND = "blend: equal weights"
INVERSE_MSE_BLEND = "blend: inverse MSE"
# A member that was never wrong would get an infinite weight
_MIN_SQUARED_ERROR = 1e-12


@dataclass(frozen=True)
class BlendStats:
    """MAE, RMSE and bias of one forecast over the paired slots of a bucket."""

    slots: int
    days: int
    mae: float
    rmse: float
    bias: float


@dataclass(frozen=True)
class BucketBlend:
    """A bucket's paired comparison of the model, the sources and the blends."""

    # The model, every source, then the blends, by label
    rows: dict[str, BlendStats]
    # Slots the inverse-MSE blend had weights for (the model alone before)
    weighted_slots: int
    # The inverse-MSE weights over every paired slot, by member; None below
    # MIN_BLEND_SLOTS
    weights: dict[str, float] | None


def inverse_mse_weights(
    squared_errors: Sequence[float], slots: int
) -> list[float] | None:
    """Return the members' weights, proportional to 1 / mean squared error.

    Args:
        squared_errors: Every member's sum of squared errors over ``slots``.
        slots: The number of slots the sums cover.

    Returns:
        Weights that sum to 1; None with fewer than ``MIN_BLEND_SLOTS`` slots.
    """
    if slots < MIN_BLEND_SLOTS:
        return None
    inverse = [slots / max(total, _MIN_SQUARED_ERROR) for total in squared_errors]
    return [value / sum(inverse) for value in inverse]


def _stats(errors: Sequence[float], days: int) -> BlendStats:
    """Return the stats of one forecast's per-slot errors."""
    return BlendStats(
        slots=len(errors),
        days=days,
        mae=statistics.fmean(abs(error) for error in errors),
        rmse=math.sqrt(statistics.fmean(error * error for error in errors)),
        bias=statistics.fmean(errors),
    )


def _blend_bucket(
    slots: Mapping[str, Mapping[str, float]], day_of: Callable[[str], date]
) -> BucketBlend | None:
    """Compare the model, the sources and the blends over a bucket's paired slots.

    Args:
        slots: Per slot (UTC key), the bucket's mean error by source.
        day_of: Returns a slot's local date.

    Returns:
        None without a source, or without a slot every member has.
    """
    sources = sorted(
        {source for errors in slots.values() for source in errors}
        - {EXTERNAL_MODEL_SOURCE}
    )
    members = [EXTERNAL_MODEL_SOURCE, *sources]
    by_day: dict[date, list[list[float]]] = {}
    for start, errors in slots.items():
        if all(member in errors for member in members):
            by_day.setdefault(day_of(start), []).append(
                [errors[member] for member in members]
            )
    if not sources or not by_day:
        return None

    paired: list[list[float]] = []
    inverse_mse: list[float] = []
    squared = [0.0] * len(members)
    weighted_slots = 0
    for day in sorted(by_day):
        # The day's weights: from the earlier days only
        weights = inverse_mse_weights(squared, len(paired))
        for row in by_day[day]:
            if weights is None:
                inverse_mse.append(row[0])
            else:
                inverse_mse.append(sum(w * e for w, e in zip(weights, row)))
                weighted_slots += 1
        paired += by_day[day]
        for index in range(len(members)):
            squared[index] += sum(row[index] ** 2 for row in by_day[day])

    days = len(by_day)
    rows = {
        member: _stats([row[index] for row in paired], days)
        for index, member in enumerate(members)
    }
    rows[EQUAL_BLEND] = _stats([statistics.fmean(row) for row in paired], days)
    rows[INVERSE_MSE_BLEND] = _stats(inverse_mse, days)
    final = inverse_mse_weights(squared, len(paired))
    return BucketBlend(
        rows=rows,
        weighted_slots=weighted_slots,
        weights=dict(zip(members, final)) if final is not None else None,
    )


def blend_stats(
    slot_errors: SlotErrors, day_of: Callable[[str], date]
) -> dict[str, BucketBlend]:
    """Return every lead-time bucket's paired comparison.

    Args:
        slot_errors: The export's ``external_slot_errors`` rows.
        day_of: Returns a slot's local date from its UTC key.

    Returns:
        The comparison per bucket that has one, in no particular order.
    """
    by_bucket: dict[str, dict[str, dict[str, float]]] = {}
    for (start, bucket), errors in slot_errors.items():
        by_bucket.setdefault(bucket, {})[start] = errors
    return {
        bucket: blend
        for bucket, slots in by_bucket.items()
        if (blend := _blend_bucket(slots, day_of)) is not None
    }
