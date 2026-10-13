"""The natural-gas price feature (#28).

Gas-fired plants often set the marginal electricity price, so the gas price
level moves the electricity price. The price is daily; a slot gets the
latest price dated **before its local day**, the price known the day
before. Training and prediction use the same rule, so a forecast days ahead
gets the latest published price, as a training row got the price known the
day before it. A price older than ``GAS_LOOKBACK_DAYS`` is not used (a
source that stopped updating gives NaN, not a stale level).

The model does not see the price itself but its **band** among the training
rows' prices (``gas_band_edges`` / ``gas_bands``, #202). The feature has one
value per day, so a tree's split on the price can set a handful of days
apart, and a price at the edge of the training window's range then gets
those days' price level whatever the weather: on 2026-10-12 a price above
all but one week of a 90-day window lifted DK1's day-ahead forecast by 5
ct/kWh. With bands of equal size every level the model can split on covers
at least ``GAS_BAND_DAYS`` days of the window, and a new extreme is in the
top band.
"""

from bisect import bisect_left
from collections.abc import Iterable
from datetime import date, timedelta
from typing import Any

import numpy as np

from ..time_slots import parse_utc
from .features import FEATURE_NAMES, optional_float

# How far back a slot looks for the latest published gas price
GAS_LOOKBACK_DAYS = 14
# The model sees the price as its band among the training rows' prices: bands
# of equal size, each covering at least GAS_BAND_DAYS days of the training
# window, GAS_MAX_BANDS at most (3 bands for 90 days, 5 from 150 days on)
GAS_BAND_DAYS = 30
GAS_MAX_BANDS = 5
_GAS_COLUMN = FEATURE_NAMES.index("gas_price")


def gas_band_edges(rows: np.ndarray, days: int) -> np.ndarray:
    """Return the band edges of the gas price in a model's training rows.

    Args:
        rows: Model input rows whose first columns are ``FEATURE_NAMES``.
        days: The days the rows cover (the training window).

    Returns:
        The quantiles that cut the rows' known gas prices into bands of
        equal size (see ``GAS_BAND_DAYS``); empty when no row has a price or
        the window is too short for two bands: every known price is then
        band 0, and the model learns nothing from it.
    """
    prices = rows[:, _GAS_COLUMN]
    known = prices[~np.isnan(prices)]
    bands = min(GAS_MAX_BANDS, days // GAS_BAND_DAYS)
    if not known.size or bands < 2:
        return np.empty(0)
    edges: np.ndarray = np.quantile(known, [band / bands for band in range(1, bands)])
    return edges


def gas_bands(rows: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Return model input rows with the gas price replaced by its band.

    Training and prediction rows alike go through this, with the edges of
    the training rows: a price below every edge is band 0, one above every
    edge (a price the training window never had) the top band. An unknown
    price stays NaN.

    Args:
        rows: Model input rows whose first columns are ``FEATURE_NAMES``.
        edges: ``gas_band_edges()`` of the model's training rows.
    """
    banded = rows.copy()
    prices = rows[:, _GAS_COLUMN]
    banded[:, _GAS_COLUMN] = np.where(
        np.isnan(prices), np.nan, np.searchsorted(edges, prices, side="right")
    )
    return banded


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
