"""Retrain scheduling: decide when the price model must be retrained.

The model is retrained when its training inputs change, not on a fixed
cadence. ``last_data_update`` is the newest of two change timestamps:

* price history — today's stored prices were added or changed (a new day,
  or tomorrow's prices arriving and extending today's entry)
* storage writes — a weather snapshot or a changed Nordpool prognosis row
  (tracked by ``LearningStorage.last_data_write``)

``predict`` retrains when the model is untrained or ``last_data_update`` is
newer than ``last_trained_at``. Hyperparameter optimization runs once per
``HPO_INTERVAL_DAYS`` new days of price data; the counter is persisted in the
``meta`` table so it survives restarts.
"""

import contextlib
import logging
from datetime import datetime
from typing import Any

import numpy as np

from homeassistant.util import dt as dt_util

from .base import PredictorBase

_LOGGER = logging.getLogger(__name__)

# New days of price data between hyperparameter optimization runs
HPO_INTERVAL_DAYS = 7


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
        self, prices: list[float], date: str | None = None
    ) -> None:
        """Store the day's prices for training and note whether they changed.

        A new date bumps the HPO counter (persisted in meta). A new date or
        changed prices mark the price history as updated, which triggers a
        retrain on this forecast run. Invalid prices (see store_daily_prices)
        are not stored and change nothing.

        Args:
            prices: Known prices for the day (today, plus tomorrow once published)
            date: Date string (YYYY-MM-DD), defaults to today
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        previous = next(
            (e.get("prices") for e in self.price_history if e.get("date") == date),
            None,
        )
        if not self.store_daily_prices(prices, date):
            return

        if previous is None:
            self._set_hpo_counter(self._hpo_counter + 1)
        elif len(previous) == len(prices) and np.allclose(
            previous, prices, rtol=0.0, atol=1e-9
        ):
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

    def retrain(self, historical_prices: list[float], features: list[dict]) -> None:
        """Train the price model and run hyperparameter optimization when due.

        ``last_trained_at`` is the time training *started*: data written while
        training runs is newer than it and triggers the next retrain.
        """
        started = dt_util.utcnow()
        self._train_models(historical_prices, features)
        if not self.is_trained:
            return
        self.last_trained_at = started

        if self._hpo_counter < HPO_INTERVAL_DAYS or len(self.price_history) < 7:
            return

        _LOGGER.info(
            "Triggering periodic hyperparameter optimization (%d new days)",
            self._hpo_counter,
        )
        if self._optimize_hyperparameters() is None:
            return
        self._set_hpo_counter(0)
        # Optimization swaps in an unfitted model with the best params — fit it
        # now so this run still predicts with the ML model
        self._train_models(historical_prices, features)

    def _set_hpo_counter(self, value: int) -> None:
        """Set the HPO day counter and persist it in the meta table."""
        self._hpo_counter = value
        with contextlib.suppress(Exception):
            self.storage.save_meta_dict({"hpo_counter": value})

    def _restore_hpo_counter(self, data: dict[str, Any]) -> None:
        """Restore the HPO day counter from loaded meta data."""
        with contextlib.suppress(TypeError, ValueError):
            self._hpo_counter = int(data.get("hpo_counter", 0))
