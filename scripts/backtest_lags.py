"""Lagged price features for the dev backtest (#119; tested, not kept).

The feature vector has no realised price in it, so #119 asked whether
origin-relative lagged prices lower the error. ``PriceLagIndex`` computes
them leak-free from the prices a model may see: every lag is relative to the
slot's **reference day**, the last day with known prices before the slot's
local day. A target row's reference day is the day before the horizon
cutoff, whatever its horizon; a training row's is the day before its own
(or, with ``mixed_ages``, 1-7 days before, hashed per day, so the training
rows show every lag age the forecast days have, next to ``price_lag_days``).

No variant lowered the MAE at every horizon in the backtest (see
docs/ml_documentation.md → Backtesting → Lagged prices), so the integration
does not have these features; ``--lags`` reproduces the experiment.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta, tzinfo
from typing import Any

import numpy as np

from custom_components.open_spot_forecast.ml.features import floor_epoch, optional_float

LAG_NAMES: tuple[str, ...] = (
    "price_same_slot_last_known_day",
    "price_mean_last_7_known_days",
    "price_same_slot_last_week",
    "price_lag_days",
)
# Days whose prices the mean level covers (the last known week)
LAG_MEAN_DAYS = 7
# ``price_same_slot_last_week`` looks back one week from the slot's day
LAG_WEEK_DAYS = 7
# Lag ages (days) mixed-age training rows show: the leads the model serves
LAG_LEADS = 7


def training_lead(day: date) -> int:
    """Return the lag age (1-``LAG_LEADS`` days) a mixed-age training row of ``day`` shows.

    A multiplicative hash of the day's ordinal spreads the ages evenly over
    the window without tying them to the weekday (any 7-day cycle would).
    """
    return 1 + (day.toordinal() * 2654435761 % 2**32) % LAG_LEADS


@dataclass(frozen=True)
class LagConfig:
    """Which lag columns a backtest model gets, and how training rows lag."""

    names: tuple[str, ...]
    mixed_ages: bool = False

    @classmethod
    def parse(cls, names: str, mixed_ages: bool) -> LagConfig:
        """Parse the ``--lags`` list (``all`` or a comma-separated subset)."""
        chosen = (
            LAG_NAMES
            if names == "all"
            else tuple(name.strip() for name in names.split(",") if name.strip())
        )
        unknown = set(chosen) - set(LAG_NAMES)
        if unknown:
            raise ValueError(
                f"unknown lag feature(s) {sorted(unknown)}; choose from {LAG_NAMES}"
            )
        return cls(chosen, mixed_ages)


class PriceLagIndex:
    """Known prices by slot, for the lagged price features (#119).

    The index only ever holds the prices visible from the origin: the
    backtest's history cut at the horizon cutoff.
    """

    def __init__(
        self, prices: Mapping[int, Any], tz: tzinfo, mixed_ages: bool = False
    ) -> None:
        """Index the known prices.

        Args:
            prices: Price per slot start (UTC epoch, as ``floor_epoch``);
                None and non-finite values are not prices.
            tz: The region's local time zone, which sets the local days.
            mixed_ages: Training rows show lag ages 1-7 (``training_lead``)
                instead of 1 day.
        """
        self._tz = tz
        self._mixed_ages = mixed_ages
        self._prices: dict[int, float] = {}
        totals: dict[date, list[float]] = {}
        for epoch, value in prices.items():
            price = optional_float(value)
            if price is None:
                continue
            self._prices[int(epoch)] = price
            day = datetime.fromtimestamp(int(epoch), tz).date()
            totals.setdefault(day, [0.0, 0.0])
            totals[day][0] += price
            totals[day][1] += 1
        self._day_totals: dict[date, tuple[float, int]] = {
            day: (total, int(count)) for day, (total, count) in totals.items()
        }
        self._means: dict[tuple[date, int], float | None] = {}
        # The local date of the latest known price; None without any
        self.last_known_day: date | None = max(totals) if totals else None

    def reference_day(self, start: datetime) -> date | None:
        """Return the slot's reference day.

        The last known day for a slot after the history (a target row,
        whatever its lead time); for a slot inside it (a training row) the
        day before, or ``training_lead`` days before with mixed ages. None
        without any prices.
        """
        if self.last_known_day is None:
            return None
        day = start.astimezone(self._tz).date()
        if day > self.last_known_day:
            return self.last_known_day
        lead = training_lead(day) if self._mixed_ages else 1
        return day - timedelta(days=lead)

    def same_slot(self, start: datetime, day: date) -> float | None:
        """Return the price of ``start``'s local wall-clock slot on ``day``."""
        local = start.astimezone(self._tz)
        moment = datetime.combine(day, local.time(), tzinfo=self._tz)
        return self._prices.get(floor_epoch(moment, self._tz))

    def mean_until(self, day: date, days: int = LAG_MEAN_DAYS) -> float | None:
        """Return the mean known price of the ``days`` local days ending on ``day``.

        None unless every one of the days has at least one known price, so
        a one-day mean never poses as a week's level.
        """
        if (day, days) in self._means:
            return self._means[day, days]
        total = 0.0
        count = 0
        for offset in range(days):
            day_total, day_count = self._day_totals.get(
                day - timedelta(days=offset), (0.0, 0)
            )
            if day_count == 0:
                self._means[day, days] = None
                return None
            total += day_total
            count += day_count
        mean = total / count
        self._means[day, days] = mean
        return mean

    def for_slot(self, start: datetime) -> dict[str, float | None]:
        """Return the slot's lagged price features (``LAG_NAMES``), None if unknown.

        The reference day's price in the slot's wall-clock slot, the mean
        price of the week ending on the reference day, the price one week
        before the slot if that day is known, and the lag's age in days.
        """
        reference = self.reference_day(start)
        if reference is None:
            return dict.fromkeys(LAG_NAMES)
        day = start.astimezone(self._tz).date()
        week_ago = day - timedelta(days=LAG_WEEK_DAYS)
        return {
            "price_same_slot_last_known_day": self.same_slot(start, reference),
            "price_mean_last_7_known_days": self.mean_until(reference),
            "price_same_slot_last_week": (
                self.same_slot(start, week_ago) if week_ago <= reference else None
            ),
            "price_lag_days": float((day - reference).days),
        }

    def columns(self, starts: np.ndarray, names: Sequence[str]) -> np.ndarray:
        """Return the lag columns ``names`` for slot starts (UTC epochs), NaN if unknown."""
        rows = []
        for start in starts.tolist():
            lags = self.for_slot(datetime.fromtimestamp(start, self._tz))
            rows.append([math.nan if lags[n] is None else lags[n] for n in names])
        return np.array(rows, dtype=float).reshape(len(rows), len(names))
