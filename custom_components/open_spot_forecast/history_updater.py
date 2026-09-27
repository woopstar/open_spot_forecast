"""Stored history for the update cycle: backfill and retention (#32, #27).

Mixed into ``ForecastUpdater``. The history the model trains on is filled in
the background at setup and after midnight: the missing price days of the
training window first (from the day-ahead APIs, whatever the displayed price
source, #24), then for every stored
price day the zone weather (Open-Meteo's archived forecasts, #23), the
Nordpool prognoses, with an ENTSO-E key the week-ahead load forecast
(#30), where it helps the gas price (#28) and, with the cross-border model,
the neighbours' prices and zone weather (#29); if anything was added, the
forecast is refreshed, so
the model retrains on it at once. The sources only request what is missing,
so an interrupted backfill resumes where it stopped. Once a day, history older than the training window plus
``HISTORY_MARGIN_DAYS`` is deleted.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

from .api.dayahead_prices import NeighbourPriceSource
from .api.openmeteo_weather import OpenMeteoWeatherSource
from .const import WEATHER_POINTS
from .ml.gas_price import GAS_LOOKBACK_DAYS
from .time_slots import local_midnight

if TYPE_CHECKING:
    from .api import NordpoolPrognosisSource
    from .api.entsoe_load import EntsoeLoadSource
    from .api.gas_prices import GasPriceSource
    from .api.time_series_source import TimeSeriesSource
    from .ml.predictor import SpotPricePredictor
    from .price_source import DayAheadPrices

_LOGGER = logging.getLogger(__name__)

# Stored history is kept this many days beyond the training window
HISTORY_MARGIN_DAYS = 2


def neighbour_sources(
    hass: HomeAssistant,
    ml_predictor: SpotPricePredictor | None,
    entsoe_api_key: str | None,
) -> tuple[list[NeighbourPriceSource], list[OpenMeteoWeatherSource]]:
    """Return the neighbours' price and zone weather sources (#29).

    Empty unless the predictor has the cross-border model. They share the
    model's database with the region's own sources.
    """
    if ml_predictor is None or ml_predictor.cross_border is None:
        return [], []
    zones = ml_predictor.cross_border.zones
    storage = ml_predictor.storage
    return (
        [NeighbourPriceSource(hass, storage, zone, entsoe_api_key) for zone in zones],
        [
            OpenMeteoWeatherSource(hass, storage, zone, neighbour=True)
            for zone in zones
            if zone in WEATHER_POINTS
        ],
    )


class HistoryUpdaterMixin:
    """Backfill and prune the stored history of a ``ForecastUpdater``."""

    hass: HomeAssistant
    entry: ConfigEntry
    api_data: dict[str, Any]
    ml_predictor: SpotPricePredictor | None
    nordpool: NordpoolPrognosisSource | None
    weather: OpenMeteoWeatherSource | None
    load: EntsoeLoadSource | None
    gas: GasPriceSource | None
    # The cross-border model's neighbour sources (#29); empty when it is off
    neighbour_prices: list[NeighbourPriceSource]
    neighbour_weather: list[OpenMeteoWeatherSource]
    dayahead: DayAheadPrices | None
    history_prices: DayAheadPrices | None

    async def refresh_forecast(self) -> None:
        """Re-read the prices and re-run the forecast (``ForecastUpdater``)."""
        raise NotImplementedError

    async def _refresh(
        self, source: TimeSeriesSource, start: datetime, end: datetime, label: str
    ) -> list[dict[str, Any]] | None:
        """Update a source and return its stored rows (``ForecastUpdater``)."""
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
        """Add the training window's missing day-ahead price days; return how many.

        Whatever the displayed price source: the day-ahead spot price is the
        model's price (#24).
        """
        if self.history_prices is None:
            return 0
        today = dt_util.now().date()
        first = today - timedelta(days=ml_predictor.max_history_days)
        try:
            days = await self.history_prices.async_history(first, today)
        except Exception as err:
            _LOGGER.warning("Day-ahead price backfill failed: %s", err)
            return 0
        if self.history_prices.license_info:
            self.api_data["price_license"] = self.history_prices.license_info
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

    async def _backfill_source(
        self, source: TimeSeriesSource | None, first: date, label: str
    ) -> bool:
        """Fetch a source's missing days from ``first`` to today; return if data changed."""
        if source is None:
            return False
        try:
            return await source.async_update(
                local_midnight(first), local_midnight(dt_util.now().date())
            )
        except Exception as err:
            _LOGGER.warning("%s history backfill failed: %s", label, err)
            return False

    async def backfill_history(self) -> None:
        """Fetch the history the model trains on that is still missing.

        Day-ahead price days first, then for the stored price days the zone
        weather, the Nordpool prognoses, the ENTSO-E load forecast, the gas
        price (#28) and the cross-border model's neighbour prices and weather
        (#29). If anything was added the forecast is refreshed, so the model
        retrains on it.
        """
        ml_predictor = self.ml_predictor
        if ml_predictor is None or self.nordpool is None:
            return
        changed = await self._backfill_prices(ml_predictor) > 0
        first = self._first_price_day()
        if first is not None:
            weather = await self._backfill_source(self.weather, first, "Open-Meteo")
            prognoses = await self._backfill_source(self.nordpool, first, "Nordpool")
            load = await self._backfill_source(self.load, first, "ENTSO-E load")
            # A slot uses the gas price known before its day (#28)
            gas = await self._backfill_source(
                self.gas, first - timedelta(days=GAS_LOOKBACK_DAYS), "Gas price"
            )
            changed = changed or weather or prognoses or load or gas
            for source in [*self.neighbour_prices, *self.neighbour_weather]:
                if await self._backfill_source(source, first, source.spec.name):
                    changed = True
        if changed:
            await self.refresh_forecast()

    async def update_gas_price(self, weather_data: dict[str, Any]) -> None:
        """Refresh the recent gas prices and attach them for the prediction (#28).

        Every slot uses the latest price published before its day, so the
        last ``GAS_LOOKBACK_DAYS`` are attached as ``weather_data["gas_price"]``.
        """
        if self.gas is None:
            return
        today = dt_util.now().date()
        rows = await self._refresh(
            self.gas,
            local_midnight(today - timedelta(days=GAS_LOOKBACK_DAYS)),
            local_midnight(today + timedelta(days=1)),
            "gas price",
        )
        if rows:
            weather_data["gas_price"] = rows

    async def update_neighbours(self, forecast_end: datetime) -> None:
        """Refresh the neighbours' recent prices and weather forecast (#29).

        Stage 1 reads both from storage, like the region's own inputs: the
        prices from yesterday to tomorrow, the weather from yesterday to
        ``forecast_end`` (Open-Meteo revises it).
        """
        today = dt_util.now().date()
        yesterday = local_midnight(today - timedelta(days=1))
        tomorrow_end = local_midnight(today + timedelta(days=2))
        updates: list[tuple[TimeSeriesSource, datetime]] = [
            *((source, tomorrow_end) for source in self.neighbour_prices),
            *((source, forecast_end) for source in self.neighbour_weather),
        ]
        for source, end in updates:
            try:
                await source.async_update(yesterday, end)
            except Exception as err:
                _LOGGER.warning("Could not update %s: %s", source.spec.name, err)

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
        weather = prices = prognoses = dayahead = zone = load = 0
        try:
            if self.history_prices is not None:
                dayahead = await self.history_prices.async_prune(cutoff)
            if ml_predictor is not None and self.nordpool is not None:
                storage = ml_predictor.storage
                weather = await self.hass.async_add_executor_job(
                    storage.delete_old_weather, keep_days
                )
                prices = await self.hass.async_add_executor_job(
                    storage.delete_old_prices, cutoff_day.isoformat()
                )
                prognoses = await self.nordpool.async_prune(cutoff)
            if self.weather is not None:
                zone = await self.weather.async_prune(cutoff)
            if self.load is not None:
                load = await self.load.async_prune(cutoff)
            if self.gas is not None:
                await self.gas.async_prune(cutoff - timedelta(days=GAS_LOOKBACK_DAYS))
            for source in [*self.neighbour_prices, *self.neighbour_weather]:
                await source.async_prune(cutoff)
        except Exception as err:
            _LOGGER.warning("Could not prune the stored history: %s", err)
            return
        _LOGGER.debug(
            "Pruned history before %s: %d weather snapshots, %d price days, "
            "%d prognosis rows, %d day-ahead prices, %d zone weather rows, "
            "%d load forecast rows",
            cutoff_day,
            weather,
            prices,
            prognoses,
            dayahead,
            zone,
            load,
        )
