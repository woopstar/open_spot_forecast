"""Stored history for the update cycle: backfill and retention (#32, #27).

Mixed into ``ForecastUpdater``. The history the model trains on is filled in
the background at setup and after midnight: with the day-ahead price source
the missing price days of the training window first (they become training
data at once), then the Nordpool prognoses for every stored price day. Both
sources only request what is missing, so an interrupted backfill resumes
where it stopped. Once a day, history older than the training window plus
``HISTORY_MARGIN_DAYS`` is deleted.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import TYPE_CHECKING

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

from .time_slots import local_midnight

if TYPE_CHECKING:
    from .api import NordpoolPrognosisSource
    from .ml.predictor import SpotPricePredictor
    from .price_source import DayAheadPrices

_LOGGER = logging.getLogger(__name__)

# Stored history is kept this many days beyond the training window
HISTORY_MARGIN_DAYS = 2


class HistoryUpdaterMixin:
    """Backfill and prune the stored history of a ``ForecastUpdater``."""

    hass: HomeAssistant
    entry: ConfigEntry
    ml_predictor: SpotPricePredictor | None
    nordpool: NordpoolPrognosisSource | None
    dayahead: DayAheadPrices | None

    async def refresh_forecast(self) -> None:
        """Re-read the prices and re-run the forecast (``ForecastUpdater``)."""
        raise NotImplementedError

    def _first_price_day(self) -> date | None:
        """Return the oldest day of the model's price history, if any."""
        days = []
        for entry in self.ml_predictor.price_history if self.ml_predictor else []:
            try:
                days.append(date.fromisoformat(str(entry.get("date"))))
            except ValueError:
                continue
        return min(days, default=None)

    async def _backfill_prices(self, ml_predictor: SpotPricePredictor) -> int:
        """Add the training window's missing day-ahead price days; return how many."""
        if self.dayahead is None:
            return 0
        today = dt_util.now().date()
        first = today - timedelta(days=ml_predictor.max_history_days)
        try:
            days = await self.dayahead.async_history(first, today)
        except Exception as err:
            _LOGGER.warning("Day-ahead price backfill failed: %s", err)
            return 0
        known = {entry.get("date") for entry in ml_predictor.price_history}
        added = 0
        for day, prices in sorted(days.items()):
            if day.isoformat() not in known:
                await self.hass.async_add_executor_job(
                    ml_predictor.record_training_prices, prices, day.isoformat()
                )
                added += 1
        if added:
            _LOGGER.info("Added %d days of day-ahead prices to the history", added)
        return added

    async def backfill_history(self) -> None:
        """Fetch the history the model trains on that is still missing.

        Day-ahead price days first (the forecast is refreshed so the model
        retrains on them), then the Nordpool prognoses for the stored price
        days up to today.
        """
        ml_predictor = self.ml_predictor
        if ml_predictor is None or self.nordpool is None:
            return
        if await self._backfill_prices(ml_predictor):
            await self.refresh_forecast()
        first = self._first_price_day()
        if first is None:
            return
        try:
            await self.nordpool.async_update(
                local_midnight(first), local_midnight(dt_util.now().date())
            )
        except Exception as err:
            _LOGGER.warning("Nordpool history backfill failed: %s", err)

    def start_history_backfill(self) -> None:
        """Run ``backfill_history`` in the background (it can take minutes)."""
        if self.nordpool is not None:
            self.entry.async_create_background_task(
                self.hass,
                self.backfill_history(),
                "open_spot_forecast_history_backfill",
            )

    async def prune_history(self) -> None:
        """Delete stored history older than the training window plus a margin.

        Without the ML model only the day-ahead prices are stored, and only
        the margin is kept.
        """
        ml_predictor = self.ml_predictor
        keep_days = HISTORY_MARGIN_DAYS
        if ml_predictor is not None:
            keep_days += ml_predictor.max_history_days
        cutoff_day = dt_util.now().date() - timedelta(days=keep_days)
        cutoff = local_midnight(cutoff_day)
        weather = prices = prognoses = dayahead = 0
        try:
            if self.dayahead is not None:
                dayahead = await self.dayahead.async_prune(cutoff)
            if ml_predictor is not None and self.nordpool is not None:
                storage = ml_predictor.storage
                weather = await self.hass.async_add_executor_job(
                    storage.delete_old_weather, keep_days
                )
                prices = await self.hass.async_add_executor_job(
                    storage.delete_old_prices, cutoff_day.isoformat()
                )
                prognoses = await self.nordpool.async_prune(cutoff)
        except Exception as err:
            _LOGGER.warning("Could not prune the stored history: %s", err)
            return
        _LOGGER.debug(
            "Pruned history before %s: %d weather snapshots, %d price days, "
            "%d prognosis rows, %d day-ahead prices",
            cutoff_day,
            weather,
            prices,
            prognoses,
            dayahead,
        )
