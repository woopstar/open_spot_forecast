"""Dated national charges from Energi Data Service (#278).

Stromligning publishes the consumer price for today and tomorrow only, so
the forecast repeats the latest known day's tariff (``TariffSchedule``).
The national charges change on known dates (1 January): Energinet's
transmission net tariff (40000), system tariff (41000) and the electricity
tax (EA-001), all owned by Energinet (``ENERGINET_GLN``). Energi Data
Service's ``DatahubPricelist`` dataset (public, no key) lists them with
``ValidFrom``/``ValidTo`` and ``Price1``..``Price24`` in DKK/kWh excl. VAT.
``Price1`` is the local hour 00-01 (Danish time); a flat rate only has
``Price1``. On a DST day the repeated hour takes the preceding hour's
price, which a lookup by local wall-clock hour gives.

``TariffChangeSource`` fetches them at most once a day, in a background
task (#291): a 429's ``Retry-After`` can hold a request for minutes, and
neither the integration setup nor the forecast waits for an enrichment.
A failed or unusable response keeps what was fetched before and logs no
URL or query.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from ..const import (
    DATAHUB_PRICELIST_API,
    DATAHUB_TZ,
    ENERGINET_GLN,
    NATIONAL_CHARGE_CODES,
)
from .http import async_get

_LOGGER = logging.getLogger(__name__)

_TZ = ZoneInfo(DATAHUB_TZ)
_TARIFF = "D03"
# A period can start years before it applies (the 2026 electricity tax is
# valid to 2028), so the query reaches back this far and is filtered locally
_LOOKBACK = timedelta(days=3 * 366)
_HOURS = 24


@dataclass(frozen=True, slots=True)
class ChargePeriod:
    """One price-list row: a charge's hourly prices over a validity period."""

    code: str
    valid_from: datetime
    valid_to: datetime | None
    prices: tuple[float | None, ...]

    def covers(self, local: datetime) -> bool:
        """Return whether the period is valid at a Danish local time."""
        return self.valid_from <= local and (
            self.valid_to is None or local < self.valid_to
        )

    def price_at(self, local: datetime) -> float | None:
        """Return the price of the local hour (``Price1`` for a flat rate)."""
        price = self.prices[local.hour]
        return self.prices[0] if price is None else price


def _local(value: object) -> datetime | None:
    """Parse a price-list timestamp (Danish local time, no offset)."""
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value).replace(tzinfo=_TZ)
    except ValueError:
        return None


def _price(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def parse_pricelist(text: str) -> list[ChargePeriod]:
    """Return the national charge periods in a ``DatahubPricelist`` response.

    Rows of other owners, types or codes, and rows without a start or a
    ``Price1``, are skipped.
    """
    try:
        records = json.loads(text).get("records")
    except AttributeError, ValueError:
        return []
    periods: list[ChargePeriod] = []
    for row in records if isinstance(records, list) else []:
        if not isinstance(row, dict) or (
            row.get("GLN_Number") != ENERGINET_GLN
            or row.get("ChargeType") != _TARIFF
            or row.get("ChargeTypeCode") not in NATIONAL_CHARGE_CODES
        ):
            continue
        valid_from = _local(row.get("ValidFrom"))
        prices = tuple(_price(row.get(f"Price{hour + 1}")) for hour in range(_HOURS))
        if valid_from is None or prices[0] is None:
            continue
        periods.append(
            ChargePeriod(
                row["ChargeTypeCode"], valid_from, _local(row.get("ValidTo")), prices
            )
        )
    return periods


class NationalCharges:
    """The sum of the national charges at any moment, from dated periods."""

    def __init__(self, periods: Iterable[ChargePeriod] = ()) -> None:
        """Initialize from the price-list periods (latest start first wins)."""
        self._periods = sorted(periods, key=lambda p: p.valid_from, reverse=True)

    def __bool__(self) -> bool:
        """Return whether any period is known."""
        return bool(self._periods)

    def total_at(self, moment: datetime) -> float | None:
        """Return the summed charges (DKK/kWh excl. VAT) at a moment.

        Args:
            moment: Timezone-aware time.

        Returns:
            The sum over ``NATIONAL_CHARGE_CODES``, each from its latest
            period valid at the moment; None if a charge has none.
        """
        local = moment.astimezone(_TZ)
        total = 0.0
        for code in NATIONAL_CHARGE_CODES:
            period = next(
                (p for p in self._periods if p.code == code and p.covers(local)),
                None,
            )
            if period is None:
                return None
            price = period.price_at(local)
            if price is None:
                return None
            total += price
        return total


class TariffChangeSource:
    """Fetches the national charges from Energi Data Service once a day."""

    def __init__(self, hass: HomeAssistant) -> None:
        """Initialize; nothing is fetched until ``async_update``."""
        self.hass = hass
        self.charges = NationalCharges()
        self._fetched_on: date | None = None
        self._task: asyncio.Task[None] | None = None

    def start_update(
        self, entry: ConfigEntry, today: date, on_update: Callable[[], None]
    ) -> None:
        """Fetch the charges in a background task of the config entry (#291).

        Nothing starts while a fetch runs or once today's charges are
        fetched. The task ends with the entry (unload cancels it).

        Args:
            entry: The config entry that owns the task.
            today: The local date (one successful fetch per day).
            on_update: Called when new charges were stored.
        """
        if self._fetched_on == today or (self._task and not self._task.done()):
            return
        self._task = entry.async_create_background_task(
            self.hass,
            self._update(today, on_update),
            "open_spot_forecast_tariff_changes",
        )

    async def _update(self, today: date, on_update: Callable[[], None]) -> None:
        """Fetch the charges; call ``on_update`` when new ones were stored."""
        if await self.async_update(today):
            on_update()

    async def async_update(self, today: date) -> bool:
        """Fetch the charges unless fetched today already.

        Returns:
            Whether new charges were stored.
        """
        if self._fetched_on == today:
            return False
        query = {
            "ChargeType": [_TARIFF],
            "GLN_Number": [ENERGINET_GLN],
            "ChargeTypeCode": list(NATIONAL_CHARGE_CODES),
        }
        response = await async_get(
            async_get_clientsession(self.hass),
            DATAHUB_PRICELIST_API,
            "Energi Data Service price list",
            params={
                "start": (today - _LOOKBACK).isoformat(),
                "filter": json.dumps(query),
                "sort": "ValidFrom desc",
                "limit": "200",
            },
        )
        periods = (
            parse_pricelist(response.text)
            if response and response.status == 200
            else []
        )
        if not periods:
            _LOGGER.warning(
                "Could not fetch the national tariffs (Energi Data Service)"
            )
            return False
        self.charges = NationalCharges(periods)
        self._fetched_on = today
        return True
