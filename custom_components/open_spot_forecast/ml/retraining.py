"""Retrain scheduling: decide when the price model must be retrained.

The model is retrained when its training inputs change, not on a fixed
cadence. ``last_data_update`` is the newest of two change timestamps:

* price history — today's stored prices were added or changed (a new day,
  or tomorrow's prices arriving and extending today's entry)
* storage writes — a weather snapshot or a changed Nordpool prognosis row
  (tracked by ``LearningStorage.last_data_write``)

``predict`` retrains when the model is untrained or ``last_data_update`` is
newer than ``last_trained_at``. The hyperparameters are the backtest-chosen
production defaults; there is no per-installation optimization (#92).
"""

from collections.abc import Sequence
from datetime import datetime

from homeassistant.util import dt as dt_util

from ..price_series import same_prices
from .base import PredictorBase


class RetrainMixin(PredictorBase):
    """Data-change tracking and retrain decisions.

    Designed to be mixed into SpotPricePredictor — all attributes
    referenced via self are provided by the owning class.
    """

    @property
    def last_data_update(self) -> datetime | None:
        """Return when training inputs last changed (UTC), or None if unknown."""
        stamps = [
            stamp
            for stamp in (self._prices_updated_at, self.storage.last_data_write)
            if stamp is not None
        ]
        return max(stamps, default=None)

    def record_training_prices(
        self, prices: Sequence[float | None], date: str | None = None
    ) -> None:
        """Store the day's prices for training and note whether they changed.

        A new date or changed prices mark the price history as updated,
        which triggers a retrain on this forecast run. Invalid prices (see store_daily_prices)
        are not stored and change nothing.

        Args:
            prices: Known prices for the day (today, plus tomorrow once published)
            date: Date string (YYYY-MM-DD), defaults to today
        """
        if date is None:
            date = dt_util.now().strftime("%Y-%m-%d")

        previous = next(
            (e.get("prices") for e in self.price_history if e.get("date") == date),
            None,
        )
        if not self.store_daily_prices(prices, date):
            return

        if previous is not None and same_prices(previous, prices):
            return

        self._prices_updated_at = dt_util.utcnow()

    def needs_retraining(self) -> bool:
        """Return True if the model is missing or its inputs changed since training."""
        if not self.is_trained or not self.price_model.trees:
            return True
        if self.last_trained_at is None:
            return True
        last_update = self.last_data_update
        return last_update is not None and last_update > self.last_trained_at

    def retrain(self) -> None:
        """Train the price model on the stored history.

        ``last_trained_at`` is the time training *started*: data written while
        training runs is newer than it and triggers the next retrain.
        """
        started = dt_util.utcnow()
        self._train_models()
        if self.is_trained:
            self.last_trained_at = started
