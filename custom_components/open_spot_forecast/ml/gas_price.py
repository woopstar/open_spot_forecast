"""The natural-gas price feature (#28).

Gas-fired plants often set the marginal electricity price, so the gas price
level moves the electricity price. The price is daily; a slot gets the
latest price dated **before its local day**, the price known the day
before. Training and prediction use the same rule, so a forecast days ahead
gets the latest published price, as a training row got the price known the
day before it. A price older than ``GAS_LOOKBACK_DAYS`` is not used (a
source that stopped updating gives NaN, not a stale level).
"""

from bisect import bisect_left
from collections.abc import Iterable
from datetime import date, timedelta
from typing import Any

from ..time_slots import parse_utc
from .features import optional_float

# How far back a slot looks for the latest published gas price
GAS_LOOKBACK_DAYS = 14


class GasPriceIndex:
    """Stored daily gas prices, for the price known before a local day."""

    def __init__(self, rows: Iterable[dict[str, Any]]) -> None:
        """Index ``gas_prices`` rows (``timestamp`` = the UTC day, ``price``)."""
        prices: dict[date, float] = {}
        for row in rows:
            moment = parse_utc(row.get("timestamp"))
            price = optional_float(row.get("price"))
            if moment is not None and price is not None:
                prices[moment.date()] = price
        self._days = sorted(prices)
        self._prices = [prices[day] for day in self._days]

    def before(self, day: date) -> float | None:
        """Return the latest price dated before ``day``; None if none is recent."""
        index = bisect_left(self._days, day)
        if index == 0:
            return None
        if self._days[index - 1] < day - timedelta(days=GAS_LOOKBACK_DAYS):
            return None
        return self._prices[index - 1]
