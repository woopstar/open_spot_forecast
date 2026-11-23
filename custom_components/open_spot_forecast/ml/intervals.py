"""Prediction intervals: a p10/p90 band around every forecast slot (#280).

The band is empirical: quantiles of the raw model's past signed errors
(``price + bias_offset - actual``, the errors the bias offsets learn from,
#203) in the prediction's lead-time bucket and local hour, over the last
``INTERVAL_WINDOW_DAYS`` scored days. With ``error = predicted - actual``
the actual price is ``predicted - error``, so

    price_low  = raw_price - q_high(errors)
    price_high = raw_price - q_low(errors)

where ``raw_price`` is the prediction before its bias offset was subtracted
and ``(q_low, q_high)`` are the ``INTERVAL_QUANTILES``. Neither bound is
clamped at 0 (prices can be negative); each is widened to the price itself
when the errors lie on one side of zero, so ``price_low <= price <=
price_high`` always holds. A group with fewer than ``MIN_INTERVAL_SAMPLES``
errors, or errors from fewer than ``MIN_INTERVAL_DAYS`` days, has no band.

The band aims at the 10th-90th percentile of the actual price. The past
errors' own 10th-90th percentile covered only 70-77 % of the next days'
prices in the backtest (the errors drift, and an hour's errors of one day
move together), so the band is their 6th-94th: 78-84 % on DK1 and DK2
(``docs/ml_documentation.md`` "Prediction Intervals").

No model is trained for it: a quantile GBM per bound would triple the
training time, and the band is calibrated on the errors the forecast
actually made. ``scripts/backtest.py --intervals`` measures the share of
actual prices inside the band per forecast day.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from datetime import date, timedelta
from typing import Any

import numpy as np

from ..const import LEAD_TIME_WINDOW_DAYS, PRICE_HIGH, PRICE_LOW
from .base import PredictorBase

_LOGGER = logging.getLogger(__name__)

# The quantiles of the past errors a band spans: aimed at 10 % of the actual
# prices below it and 10 % above (p10/p90). Their own 10th-90th percentile
# covered 70-77 % in the backtest, the 6th-94th 78-84 % (#280)
INTERVAL_QUANTILES = (0.06, 0.94)
# Days of scored errors a band is computed from (the lead-time accuracy's)
INTERVAL_WINDOW_DAYS = LEAD_TIME_WINDOW_DAYS
# A (bucket, hour) group with fewer errors, or errors of fewer days, has no
# band: the errors of one day move together
MIN_INTERVAL_SAMPLES = 50
MIN_INTERVAL_DAYS = 7

type IntervalKey = tuple[str, int]


def interval_key(bucket: str, slot: int) -> IntervalKey:
    """Return the group a slot's errors are pooled in: lead-time bucket and local hour.

    Args:
        bucket: ``LEAD_TIME_BUCKETS`` key of the prediction.
        slot: 15-minute slot of the local day by wall clock (0-95).
    """
    return bucket, slot // 4


def error_quantiles(errors: Sequence[float]) -> tuple[float, float] | None:
    """Return the ``INTERVAL_QUANTILES`` of signed errors (predicted - actual).

    Args:
        errors: One group's errors.

    Returns:
        ``(q_low, q_high)``, or None with fewer than ``MIN_INTERVAL_SAMPLES``.
    """
    if len(errors) < MIN_INTERVAL_SAMPLES:
        return None
    low, high = np.quantile(np.asarray(errors, dtype=float), INTERVAL_QUANTILES)
    return float(low), float(high)


def interval_bounds(
    price: float, raw_price: float, quantiles: tuple[float, float] | None
) -> tuple[float, float] | None:
    """Return a prediction's ``(low, high)`` bounds from its group's error quantiles.

    Args:
        price: The prediction as served (bias offset subtracted).
        raw_price: The model's prediction before the offset.
        quantiles: ``(q_low, q_high)`` of the group's raw errors, or None.

    Returns:
        ``(raw_price - q_high, raw_price - q_low)``, widened to include
        ``price``; never clamped at 0. None without quantiles.
    """
    if quantiles is None:
        return None
    q_low, q_high = quantiles
    return min(raw_price - q_high, price), max(raw_price - q_low, price)


def group_quantiles(
    rows: Iterable[tuple[str, str, int, float]],
) -> dict[IntervalKey, tuple[float, float]]:
    """Return every group's error quantiles from ``(date, bucket, slot, error)`` rows.

    Groups with too few errors (see ``error_quantiles``) or with errors of
    fewer than ``MIN_INTERVAL_DAYS`` dates are left out.
    """
    groups: dict[IntervalKey, list[float]] = {}
    days: dict[IntervalKey, set[str]] = {}
    for day, bucket, slot, error in rows:
        key = interval_key(bucket, slot)
        groups.setdefault(key, []).append(error)
        days.setdefault(key, set()).add(day)
    quantiles: dict[IntervalKey, tuple[float, float]] = {}
    for key, errors in groups.items():
        if len(days[key]) < MIN_INTERVAL_DAYS:
            continue
        if (pair := error_quantiles(errors)) is not None:
            quantiles[key] = pair
    return quantiles


def with_bounds(prediction: dict[str, Any], bounds: tuple[float, float] | None) -> None:
    """Add ``price_low``/``price_high`` to a prediction in place, if it has bounds."""
    if bounds is not None:
        prediction[PRICE_LOW], prediction[PRICE_HIGH] = bounds


class IntervalMixin(PredictorBase):
    """The error quantiles behind the forecast's p10/p90 bounds."""

    def record_interval_errors(
        self, slot_date: str, slot: int, errors: Mapping[str, Sequence[float]]
    ) -> None:
        """Store a scored slot's raw errors per lead-time bucket (blocking).

        Args:
            slot_date: The slot's local date (``YYYY-MM-DD``).
            slot: The slot of the local day by wall clock (0-95).
            errors: Raw errors (predicted + offset - actual) per bucket.
        """
        self.storage.add_interval_errors(slot_date, slot, errors)

    def refresh_interval_quantiles(self, today: date) -> None:
        """Prune errors outside the window and reload the quantiles (blocking).

        Runs before every forecast run; a storage failure is logged and
        leaves the forecast without bounds rather than with stale ones.

        Args:
            today: The local date of the forecast run.
        """
        cutoff = (today - timedelta(days=INTERVAL_WINDOW_DAYS - 1)).isoformat()
        try:
            self.storage.delete_interval_errors_before(cutoff)
            self.interval_quantiles = group_quantiles(
                self.storage.get_interval_errors(cutoff)
            )
        except sqlite3.Error as err:
            _LOGGER.error("Failed to load the prediction interval errors: %s", err)
            self.interval_quantiles = {}

    def prediction_interval(
        self, price: float, offset: float, slot: int, bucket: str
    ) -> tuple[float, float] | None:
        """Return the p10/p90 bounds of a prediction, or None without enough errors.

        Args:
            price: The prediction as served (bias offset subtracted).
            offset: The bias offset that was subtracted.
            slot: The slot of the local day by wall clock (0-95).
            bucket: The prediction's lead-time bucket.
        """
        return interval_bounds(
            price,
            price + offset,
            self.interval_quantiles.get(interval_key(bucket, slot)),
        )
