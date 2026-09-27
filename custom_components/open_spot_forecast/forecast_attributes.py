"""How a forecast is laid out in entity attributes (#38).

``detailed`` (the default) is one dict per interval: ``start``, ``end``,
``price``, ``unit`` and ``confidence``, about 120 bytes each. ``compact`` is
EpexPredictor's short format plus confidence: parallel arrays of the
intervals' unix start seconds (``s``), prices (``t``) and confidences in
percent (``c``, like the Prediction Confidence sensor), with the unit and the
interval length once. That is about 21 bytes per 15-minute slot, so a week
(168 hours, 672 slots) fits Home Assistant's 16 KB limit.

The recorder does not store the attributes of a state whose JSON exceeds
``RECORDER_MAX_ATTRIBUTES_BYTES``. ``fit_compact()`` trims whole hours from
the end of a compact forecast until the attributes fit, whatever the unit,
precision or prices.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from homeassistant.helpers.json import json_bytes

from .const import RECORDER_MAX_ATTRIBUTES_BYTES
from .price_output import HOUR_MINUTES
from .time_slots import parse_utc

# What an entity's own attributes may use; the rest is left for the ones Home
# Assistant adds (friendly name, icon, unit, device class, attribution)
ATTRIBUTE_BUDGET_BYTES = RECORDER_MAX_ATTRIBUTES_BYTES - 1024

# The arrays of a compact forecast
COMPACT_ARRAYS = ("s", "t", "c")


def detailed_forecast(entries: Sequence[dict[str, Any]], unit: str) -> list[dict]:
    """Return forecast entries as the detailed attribute: one dict each.

    Args:
        entries: ``{"start", "end", "price", "confidence"}`` per interval.
        unit: Unit of the prices, repeated in every entry.
    """
    return [{**entry, "unit": unit} for entry in entries]


def compact_forecast(
    entries: Sequence[dict[str, Any]], unit: str, interval_minutes: int
) -> dict[str, Any]:
    """Return forecast entries as the compact attribute: parallel arrays.

    Args:
        entries: ``{"start", "price", "confidence"}`` per interval, in order;
            one without a parseable ``start`` is skipped.
        unit: Unit of the prices.
        interval_minutes: Length of one interval (15, or 60 when hourly).

    Returns:
        ``interval_minutes``, ``unit``, ``s`` (unix start seconds), ``t``
        (prices) and ``c`` (confidence in percent, None if unknown).
    """
    rows = [
        (start, entry)
        for entry in entries
        if (start := parse_utc(entry["start"])) is not None
    ]
    return {
        "interval_minutes": interval_minutes,
        "unit": unit,
        "s": [int(start.timestamp()) for start, _ in rows],
        "t": [entry["price"] for _, entry in rows],
        "c": [
            None
            if entry.get("confidence") is None
            else round(entry["confidence"] * 100)
            for _, entry in rows
        ],
    }


def attributes_size(attributes: dict[str, Any]) -> int:
    """Return the size of attributes as the recorder stores them (JSON bytes)."""
    return len(json_bytes(attributes))


def fit_compact(
    attributes: dict[str, Any],
    key: str,
    budget: int = ATTRIBUTE_BUDGET_BYTES,
) -> None:
    """Trim whole hours from the end of a compact forecast until attributes fit.

    Args:
        attributes: The entity's attributes; ``attributes[key]`` is a
            ``compact_forecast()`` and is trimmed in place.
        key: The attribute holding the compact forecast.
        budget: The largest allowed JSON size of ``attributes``.
    """
    compact = attributes[key]
    per_hour = max(1, HOUR_MINUTES // compact["interval_minutes"])
    while (size := attributes_size(attributes)) > budget and compact["s"]:
        count = len(compact["s"])
        per_entry = max(1, attributes_size(compact) // count)
        drop = math.ceil(math.ceil((size - budget) / per_entry) / per_hour) * per_hour
        keep = max(0, count - drop)
        for name in COMPACT_ARRAYS:
            compact[name] = compact[name][:keep]
