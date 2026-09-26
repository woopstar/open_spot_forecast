"""The configured price source and the day-ahead price path (#27).

``PriceSettings`` reads a config entry's price source (Stromligning's sensors
or day-ahead auction prices), ENTSO-E token, currency and VAT.
``DayAheadPrices`` is what the update cycle uses for the day-ahead source: it
keeps the stored prices current (``DayAheadPriceSource``), converts them into
the configured currency (``ExchangeRates``) and returns today's and
tomorrow's prices in the shape ``SensorReader.read_spot_prices()`` returns,
plus older days for the model's price history.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant

from .api.dayahead_prices import DayAheadPriceSource
from .api.exchange_rates import ExchangeRates
from .const import (
    CONF_CURRENCY,
    CONF_ENTSOE_API_KEY,
    CONF_PRICE_SOURCE,
    CONF_VAT,
    DEFAULT_CURRENCY,
    DEFAULT_PRICE_SOURCE,
    DEFAULT_VAT,
    PRICE_SOURCE_DAYAHEAD,
)
from .spot_prices import dayahead_prices_by_day, dayahead_spot_data
from .time_slots import local_midnight

if TYPE_CHECKING:
    from .ml.storage import LearningStorage


@dataclass(frozen=True, slots=True)
class PriceSettings:
    """Where a config entry's prices come from, and how they are shown.

    Options (reconfiguration) take precedence over the entry's initial data.
    """

    source: str
    currency: str
    vat: float
    # Never shown in a repr or log
    entsoe_api_key: str | None = field(default=None, repr=False)

    @classmethod
    def from_entry(cls, entry: ConfigEntry) -> PriceSettings:
        """Read the price settings of a config entry."""

        def option(key: str, default: Any) -> Any:
            return entry.options.get(key, entry.data.get(key, default))

        return cls(
            source=str(option(CONF_PRICE_SOURCE, DEFAULT_PRICE_SOURCE)),
            currency=str(option(CONF_CURRENCY, DEFAULT_CURRENCY)),
            vat=float(option(CONF_VAT, DEFAULT_VAT)),
            entsoe_api_key=option(CONF_ENTSOE_API_KEY, None) or None,
        )

    @property
    def dayahead(self) -> bool:
        """Return whether prices come from the day-ahead APIs."""
        return self.source == PRICE_SOURCE_DAYAHEAD


def with_vat(prices: list[float | None], vat: float) -> list[float | None]:
    """Return spot prices with VAT added (missing slots stay None)."""
    return [None if price is None else price * (1 + vat) for price in prices]


class DayAheadPrices:
    """Day-ahead prices for the update cycle: fetched, stored, converted."""

    def __init__(
        self,
        hass: HomeAssistant,
        storage: LearningStorage,
        region: str,
        settings: PriceSettings,
    ) -> None:
        """Initialize the price path for a region; nothing is fetched yet."""
        self.source = DayAheadPriceSource(
            hass, storage, region, settings.entsoe_api_key
        )
        self.rates = ExchangeRates(hass, settings.currency)

    @property
    def license_info(self) -> str | None:
        """Return the licence of the latest energy-charts response."""
        return self.source.license_info

    async def async_read(self, today: date) -> dict[str, list]:
        """Fetch what is missing and return today's and tomorrow's prices.

        Returns:
            ``read_spot_prices()``-shaped data in currency/kWh excl. VAT;
            ``today`` is empty if today's prices are not known.
        """
        start = local_midnight(today)
        end = local_midnight(today + timedelta(days=2))
        await self.source.async_update(start, end)
        await self.rates.async_update(today, today)
        rows = await self.source.async_load(start, end)
        return dayahead_spot_data(rows, self.rates.rate, today)

    async def async_history(
        self, first: date, today: date
    ) -> dict[date, list[float | None]]:
        """Fetch the missing days from ``first`` to yesterday and return them.

        Returns:
            Prices (currency/kWh excl. VAT, one per slot) by local day, for
            the days that have valid prices and a rate.
        """
        start = local_midnight(first)
        end = local_midnight(today)
        await self.source.async_update(start, end)
        await self.rates.async_update(first, today)
        rows = await self.source.async_load(start, end)
        return dayahead_prices_by_day(rows, self.rates.rate)

    async def async_prune(self, before: datetime) -> int:
        """Delete stored prices older than ``before``."""
        return await self.source.async_prune(before)
