"""Yesterday's price in the same slot, for the near-term price model (#135).

The canonical feature vector holds no realised price: one model serves
forecast days 1-7, and a lagged price it learned from one-day-old training
rows would be up to seven days old at prediction (#119). The **near-term
model** is a second price model for the first ``NEAR_TERM_DAYS`` forecast
days only, with one more column, ``NEAR_TERM_FEATURE``: the price in the
slot's local wall-clock slot on its **reference day**, the last day with
known prices before the slot's local day.

* A training row's reference day is the day before its own.
* A prediction row's is the last known day, whatever its lead time, so it
  never sees a price from its own day or later.

``PriceLagIndex`` computes the column for both; the dev backtest
(``scripts/backtest_lags.py``) extends it with the lag variants #119 tried.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime, timedelta, tzinfo
from typing import Any

import numpy as np

from ..time_slots import slot_start_in_day
from .features import FEATURE_NAMES, floor_epoch, optional_float

# Forecast days (counted from the last day with known prices) the near-term
# model serves, in the regions that have it (``NEAR_TERM_REGIONS`` in
# const.py); the plain price model serves the later ones. Chosen with the
# backtest (docs/ml_documentation.md → Near-Term Model)
NEAR_TERM_DAYS = 2
# The near-term model's extra column, after the canonical features
NEAR_TERM_FEATURE = "price_same_slot_last_known_day"


class PriceLagIndex:
    """Known prices by slot, for the lagged price of the near-term model."""

    def __init__(self, prices: Mapping[int, Any], tz: tzinfo) -> None:
        """Index the known prices.

        Args:
            prices: Price per slot start (UTC epoch, as ``floor_epoch``);
                None and non-finite values are not prices.
            tz: The region's local time zone, which sets the local days.
        """
        self._tz = tz
        self._prices: dict[int, float] = {}
        for epoch, value in prices.items():
            price = optional_float(value)
            if price is not None:
                self._prices[int(epoch)] = price
        # The local date of the latest known price; None without any
        self.last_known_day: date | None = (
            datetime.fromtimestamp(max(self._prices), tz).date()
            if self._prices
            else None
        )

    @classmethod
    def from_days(cls, days: Iterable[Mapping[str, Any]], tz: tzinfo) -> PriceLagIndex:
        """Index the model's price history (``{"date", "prices"}`` per local day).

        A day's list runs on the 15-minute grid from its local midnight and
        may reach into the next day (today's entry holds tomorrow's prices
        once published); every slot is keyed by its own UTC start.
        """
        prices: dict[int, Any] = {}
        for entry in days:
            try:
                day = date.fromisoformat(str(entry.get("date")))
            except ValueError:
                continue
            for index, price in enumerate(entry.get("prices") or []):
                prices[floor_epoch(slot_start_in_day(day, index, tz), tz)] = price
        return cls(prices, tz)

    def _training_lead(self, day: date) -> int:
        """Return the lag age in days of a training row of ``day``."""
        return 1

    def reference_day(self, start: datetime) -> date | None:
        """Return the slot's reference day.

        The last known day for a slot after the history (a prediction row,
        whatever its lead time); the day before for a slot inside it (a
        training row). None without any prices.
        """
        if self.last_known_day is None:
            return None
        day = start.astimezone(self._tz).date()
        if day > self.last_known_day:
            return self.last_known_day
        return day - timedelta(days=self._training_lead(day))

    def lag_days(self, start: datetime) -> int | None:
        """Return the age in days of the slot's lag: its day minus its reference day."""
        reference = self.reference_day(start)
        if reference is None:
            return None
        return (start.astimezone(self._tz).date() - reference).days

    def same_slot(self, start: datetime, day: date) -> float | None:
        """Return the price of ``start``'s local wall-clock slot on ``day``."""
        local = start.astimezone(self._tz)
        moment = datetime.combine(day, local.time(), tzinfo=self._tz)
        return self._prices.get(floor_epoch(moment, self._tz))

    def column(self, starts: Sequence[datetime | None]) -> np.ndarray:
        """Return ``NEAR_TERM_FEATURE`` for each slot start, NaN if unknown."""
        values = np.full(len(starts), np.nan)
        for row, start in enumerate(starts):
            reference = None if start is None else self.reference_day(start)
            if start is None or reference is None:
                continue
            price = self.same_slot(start, reference)
            if price is not None:
                values[row] = price
        return values

    def near_term(self, starts: Sequence[datetime | None], days: int) -> np.ndarray:
        """Return which slots the near-term model serves: a lag at most ``days`` old.

        A slot without a start time or a reference day is not one of them.
        """
        ages = [None if start is None else self.lag_days(start) for start in starts]
        return np.array([age is not None and age <= days for age in ages], dtype=bool)


def slot_start(feature: Mapping[str, Any]) -> datetime | None:
    """Return a feature row's slot start, or None if it has none that parses."""
    try:
        return datetime.fromisoformat(str(feature.get("start")))
    except ValueError:
        return None


def near_term_inputs(rows: np.ndarray, lag: np.ndarray) -> np.ndarray:
    """Return the near-term model's rows: ``lag`` after the canonical features.

    Args:
        rows: The price model's input rows (the canonical features, then
            any cross-border columns).
        lag: ``NEAR_TERM_FEATURE`` per row (``PriceLagIndex.column``).
    """
    return np.insert(rows, len(FEATURE_NAMES), lag, axis=1)
