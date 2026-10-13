"""The natural-gas price feature (#28).

Gas-fired plants often set the marginal electricity price, so the gas price
level moves the electricity price. The price is daily; a slot gets the
**mean of the last ``GAS_LEVEL_PRICES`` prices dated before its local day**:
the level known the day before. Training and prediction use the same rule,
so a forecast days ahead gets the latest known level, as a training row got
the level known the day before it. A latest price older than
``GAS_LOOKBACK_DAYS`` gives no level (a source that stopped updating gives
NaN, not a stale level).

The level, not the latest price (#202): the feature has one value per day,
so a tree's split on it sets a handful of days apart, and a price that
jumps to the edge of the training window's range gets those days' price
level whatever the weather. On 2026-10-12 a jump from 324 to 373 PLN/MWh
lifted DK1's day-ahead forecast by 5 ct/kWh. A mean of 7 prices moves by a
seventh of a jump per day.
"""

from bisect import bisect_left
from collections.abc import Iterable
from datetime import date, timedelta
from typing import Any

from ..time_slots import parse_utc
from .features import optional_float

# How far back a slot looks for the latest published gas price
GAS_LOOKBACK_DAYS = 14
# The level is the mean of this many published prices (fewer if fewer are
# known inside the lookback)
GAS_LEVEL_PRICES = 7


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
        """Return the price level known before ``day``; None if none is recent.

        The mean of the last ``GAS_LEVEL_PRICES`` prices dated before
        ``day`` and within ``GAS_LOOKBACK_DAYS`` of it.
        """
        index = bisect_left(self._days, day)
        oldest = bisect_left(self._days, day - timedelta(days=GAS_LOOKBACK_DAYS))
        recent = self._prices[max(oldest, index - GAS_LEVEL_PRICES) : index]
        return sum(recent) / len(recent) if recent else None
