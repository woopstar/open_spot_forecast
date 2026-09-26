"""ECB euro reference rates, to convert EUR day-ahead prices (#27).

Day-ahead prices are published in EUR/MWh. A region priced in another
currency (DKK, SEK, NOK) converts a day's prices with the ECB reference rate
of that day, or the latest one before it: the ECB publishes on TARGET
business days, and tomorrow's prices come before tomorrow's rate. Rates are
fetched from the ECB data API at most once a day, for the days needed. If
the ECB cannot be reached, DKK (pegged to the euro in ERM II) falls back to
its central rate; other currencies have no rate until the ECB answers.
"""

import csv
import io
import logging
from datetime import date, timedelta

from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from ..const import ECB_RATES_API
from .http import async_get

_LOGGER = logging.getLogger(__name__)

# ERM II central rates, used only while the ECB has not answered
PEGGED_EUR_RATES = {"DKK": 7.46038}
# Fetched before the first day needed, so a weekend or holiday has a rate
_LOOKBACK = timedelta(days=7)


def parse_ecb_csv(text: str) -> dict[date, float]:
    """Return the daily rates in an ECB ``csvdata`` response."""
    rates: dict[date, float] = {}
    for row in csv.DictReader(io.StringIO(text)):
        try:
            rates[date.fromisoformat(row["TIME_PERIOD"])] = float(row["OBS_VALUE"])
        except KeyError, TypeError, ValueError:
            continue
    return rates


class ExchangeRates:
    """EUR to one currency, by day."""

    def __init__(self, hass: HomeAssistant, currency: str) -> None:
        """Initialize for a currency (``EUR`` needs no rates)."""
        self.hass = hass
        self.currency = currency
        self._rates: dict[date, float] = {}
        self._fetched_on: date | None = None
        self._fetched_from: date | None = None

    async def async_update(self, first_day: date, today: date) -> None:
        """Fetch the rates from ``first_day`` on, unless fetched today already."""
        if self.currency == "EUR":
            return
        start = first_day - _LOOKBACK
        if (
            self._fetched_on == today
            and self._fetched_from is not None
            and self._fetched_from <= start
        ):
            return
        response = await async_get(
            async_get_clientsession(self.hass),
            f"{ECB_RATES_API}/D.{self.currency}.EUR.SP00.A",
            "ECB exchange rate",
            params={"startPeriod": start.isoformat(), "format": "csvdata"},
        )
        rates = (
            parse_ecb_csv(response.text) if response and response.status == 200 else {}
        )
        if not rates:
            _LOGGER.warning("Could not fetch the EUR/%s exchange rate", self.currency)
            return
        self._rates.update(rates)
        self._fetched_on = today
        self._fetched_from = start

    def rate(self, day: date) -> float | None:
        """Return EUR to the currency on ``day`` (the latest rate on or before it)."""
        if self.currency == "EUR":
            return 1.0
        earlier = [known for known in self._rates if known <= day]
        if earlier:
            return self._rates[max(earlier)]
        return PEGGED_EUR_RATES.get(self.currency)
