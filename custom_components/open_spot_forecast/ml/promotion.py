"""Promotion gate of a retrained price model (#275).

A retrain fits a candidate into new model objects. The candidate replaces
the live models only if its holdout error is not clearly worse than the live
model's and its holdout predictions look sane. Otherwise the live models
keep serving, so one bad data day (a stuck sensor, a partial weather
archive, a broken prognosis) cannot degrade the forecast until the next
retrain.

The gate (``rejection_reason``) compares the candidate's holdout MAE with
the live model's, from its own training. The two are scored on different
rows (the newest 20 % moves with every retrain), so the gate has a wide
margin: a candidate is rejected when its MAE exceeds
``max(previous * PROMOTION_MAE_FACTOR, previous + PROMOTION_MAE_FLOOR)``.
Without a live model (the first training, or the first after a restart)
the candidate is accepted unless its predictions are not finite. So is
every candidate while the history is shorter than ``PROMOTION_MIN_ROWS``:
with a few days, each retrain moves the holdout to mostly new days and the
two MAEs say nothing about each other.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .cross_border import Stage1Model
    from .gbm import NumpyGradientBoosting

# A candidate may score this much worse than the live model's holdout MAE
PROMOTION_MAE_FACTOR = 1.25
# ... or this much (raw spot price excl. VAT, currency/kWh), whichever is
# larger, so a tiny previous MAE does not reject every candidate
PROMOTION_MAE_FLOOR = 0.005
# The gate compares MAEs from 14 days of history on (a holdout of 3 days)
PROMOTION_MIN_ROWS = 14 * 96
# Holdout predictions whose spread is below this share of the actual
# prices' spread are near constant: the model learned nothing
NEAR_CONSTANT_RATIO = 0.01

# Rejection reasons, stored in meta (``last_retrain_rejected_reason``)
REJECTED_NON_FINITE = "non_finite"
REJECTED_NEAR_CONSTANT = "near_constant"
REJECTED_WORSE_MAE = "worse_mae"
REJECTED_ERROR = "error"


@dataclass(frozen=True)
class ModelSnapshot:
    """The live models and what they were trained with, kept for a rollback.

    Training refits the gas price bands and the cross-border stage-1 models
    before the candidate is scored, so those are kept too.
    """

    price_model: NumpyGradientBoosting
    near_model: NumpyGradientBoosting
    gas_edges: np.ndarray
    training_samples: int
    stage1: dict[str, Stage1Model] | None
    stage1_fit_seconds: float | None


def promotion_limit(previous_mae: float) -> float:
    """Return the highest holdout MAE a candidate may have to be promoted."""
    return max(previous_mae * PROMOTION_MAE_FACTOR, previous_mae + PROMOTION_MAE_FLOOR)


def previous_mae_to_beat(
    had_model: bool, previous_mae: float | None, rows: int
) -> float | None:
    """Return the live model's holdout MAE the candidate is gated on, or None.

    None without a live model, without its MAE, or with fewer than
    ``PROMOTION_MIN_ROWS`` training rows (before the masked copies).
    """
    if not had_model or rows < PROMOTION_MIN_ROWS:
        return None
    return previous_mae


def rejection_reason(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    candidate_mae: float,
    previous_mae: float | None,
) -> str | None:
    """Return why a candidate must not be promoted, or None to promote it.

    Args:
        y_true: The holdout rows' actual prices.
        y_pred: The candidate's holdout copy's predictions for them.
        candidate_mae: The candidate's holdout MAE.
        previous_mae: The live model's holdout MAE; None without a live
            model or below ``PROMOTION_MIN_ROWS`` (see ``previous_mae_to_beat``),
            which only rejects non-finite predictions.

    Returns:
        ``REJECTED_NON_FINITE`` for a non-finite prediction or MAE,
        ``REJECTED_NEAR_CONSTANT`` when the predictions barely vary while
        the prices do, ``REJECTED_WORSE_MAE`` above ``promotion_limit``.
    """
    if not math.isfinite(candidate_mae) or not np.isfinite(y_pred).all():
        return REJECTED_NON_FINITE
    if previous_mae is None:
        return None
    spread = float(np.std(y_true)) if len(y_true) else 0.0
    if spread > 1e-9 and float(np.std(y_pred)) < NEAR_CONSTANT_RATIO * spread:
        return REJECTED_NEAR_CONSTANT
    if candidate_mae > promotion_limit(previous_mae):
        return REJECTED_WORSE_MAE
    return None
