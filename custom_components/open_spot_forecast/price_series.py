"""Validation of a day's price series before it is stored, trained on or learned from."""

import math
from collections.abc import Sequence

# A price this close to 0 counts as zero (floats are never compared with ==)
_ZERO_PRICE_EPSILON = 1e-9


def is_invalid_price_series(prices: Sequence[float | None]) -> bool:
    """Return True if a day's prices must not be used.

    A failing price source (a sensor back from a restart, an API hiccup)
    often reports 0 for every slot. A day is invalid when every price is
    zero, or when any price is missing (None) or not finite (NaN, inf).
    Some zero or negative prices are normal and keep a day valid. An empty
    series means "no data" rather than bad data, so it is not invalid.

    Args:
        prices: The day's prices, in any order.

    Returns:
        True if the series is non-empty and all zero, or has a missing or
        non-finite price.
    """
    if not prices:
        return False
    values: list[float] = []
    for price in prices:
        if price is None or not math.isfinite(price):
            return True
        values.append(price)
    return all(abs(value) <= _ZERO_PRICE_EPSILON for value in values)
