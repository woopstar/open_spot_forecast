"""What the user pays: the one transformation of every exposed price (#39).

The model predicts, and the day-ahead source delivers, the raw spot price in
currency/kWh excl. VAT and tariffs (#16). Every spot-based price an entity
exposes goes through ``PriceOutput``, once:

    total = (spot + surcharge) × (1 + VAT)

in the configured unit (kWh, MWh or Wh), with the surcharge in the currency
per that unit. Stromligning's consumer prices already include tariffs, VAT
and the supplier's surcharge ("all-in"), so they are only converted to the
unit. With ``hourly_average`` every series is averaged per local hour first,
for contracts billed by the hour.

Averaging, then converting, gives the same result as converting each price
first: the conversion is affine and increasing, so it also keeps the order
of prices (min/max) and commutes with the mean.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from homeassistant.util import dt as dt_util

from .const import (
    DEFAULT_HOURLY_AVERAGE,
    DEFAULT_PRECISION,
    DEFAULT_PRICE_TYPE,
    DEFAULT_SURCHARGE,
    DEFAULT_VAT,
    PRICE_IN,
    SLOTS_PER_HOUR,
    SOURCE_ACTUAL,
    SOURCE_PREDICTED,
)
from .time_slots import SLOT_MINUTES, floor_to_slot, parse_utc, slot_index_in_day

HOUR_MINUTES = 60


def apply_price_components(spot: float, surcharge: float, vat: float) -> float:
    """Return the price paid for a spot price: ``(spot + surcharge) × (1 + vat)``.

    Args:
        spot: Spot price excl. VAT, in the currency per the output unit.
        surcharge: Fixed amount added per unit before VAT.
        vat: VAT rate as a fraction (0.25 for 25 %).

    Returns:
        The total price, unrounded.
    """
    return (spot + surcharge) * (1 + vat)


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def hourly_averages(prices: Sequence[float | None]) -> list[float | None]:
    """Average a local day's 15-minute prices per local hour.

    The slots of a day are stepped on the UTC timeline from local midnight
    (``time_slots``) and every zone's offset is a whole hour, so each local
    hour is four consecutive slots: 23, 24 or 25 hours on DST days.

    Args:
        prices: One price per slot from local midnight, None for a missing one.

    Returns:
        One value per local hour: the mean of the hour's known prices, None
        if the hour has none.
    """
    return [
        _mean([price for price in prices[i : i + SLOTS_PER_HOUR] if price is not None])
        for i in range(0, len(prices), SLOTS_PER_HOUR)
    ]


def _predicted_slots(
    predictions: Iterable[dict[str, Any]],
) -> Iterator[tuple[datetime, dict[str, Any]]]:
    """Yield (UTC start, prediction) for the predictions with a start and a price."""
    for prediction in predictions:
        start = parse_utc(prediction.get("start"))
        if start is not None and prediction.get("price") is not None:
            yield start, prediction


def _with_source(
    entry: dict[str, Any], slots: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """Mark an entry ``actual`` or ``predicted`` if its slots are marked (#40).

    An hour is ``actual`` only if all its slots are.
    """
    sources = {slot["source"] for slot in slots if "source" in slot}
    if sources:
        entry["source"] = (
            SOURCE_ACTUAL if sources == {SOURCE_ACTUAL} else SOURCE_PREDICTED
        )
    return entry


def slot_forecast(predictions: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the predictions that have a start and a price, one per slot.

    Args:
        predictions: Model predictions with ISO ``start``/``end``, ``price``
            (raw spot) and ``confidence``.

    Returns:
        ``{"start", "end", "price", "confidence"}`` entries; ``end`` is filled
        in (start + 15 minutes) when a prediction has none. A prediction's
        ``source`` (#40) is kept.
    """
    return [
        _with_source(
            {
                "start": prediction["start"],
                "end": prediction.get("end")
                or dt_util.as_local(
                    start + timedelta(minutes=SLOT_MINUTES)
                ).isoformat(),
                "price": prediction["price"],
                "confidence": prediction.get("confidence"),
            },
            [prediction],
        )
        for start, prediction in _predicted_slots(predictions)
    ]


def hourly_forecast(predictions: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Average predictions per hour (on the UTC timeline, so DST-safe).

    Args:
        predictions: Model predictions (see ``slot_forecast``).

    Returns:
        One entry per hour that has a prediction, in order: the hour's local
        ``start``/``end``, the mean ``price`` of its predicted slots and
        their mean ``confidence`` (None without one).
    """
    hours: dict[datetime, list[dict[str, Any]]] = {}
    for start, prediction in _predicted_slots(predictions):
        hours.setdefault(floor_to_slot(start, HOUR_MINUTES), []).append(prediction)
    forecast = []
    for hour_start, slots in hours.items():
        confidence = _mean(
            [slot["confidence"] for slot in slots if slot.get("confidence") is not None]
        )
        forecast.append(
            _with_source(
                {
                    "start": dt_util.as_local(hour_start).isoformat(),
                    "end": dt_util.as_local(
                        hour_start + timedelta(minutes=HOUR_MINUTES)
                    ).isoformat(),
                    "price": _mean([slot["price"] for slot in slots]),
                    "confidence": None if confidence is None else round(confidence, 2),
                },
                slots,
            )
        )
    return forecast


@dataclass(frozen=True, slots=True)
class PriceOutput:
    """How a config entry exposes prices: unit, surcharge, VAT, rounding, hourly.

    Read from the entry by ``PriceSettings.from_entry()``; the defaults are
    those of an install that never set the options (kWh, no surcharge,
    15-minute values).
    """

    vat: float = DEFAULT_VAT
    surcharge: float = DEFAULT_SURCHARGE
    price_type: str = DEFAULT_PRICE_TYPE
    precision: int = DEFAULT_PRECISION
    hourly_average: bool = DEFAULT_HOURLY_AVERAGE

    @property
    def interval_minutes(self) -> int:
        """Return the length of one exposed price interval in minutes."""
        return HOUR_MINUTES if self.hourly_average else SLOT_MINUTES

    def unit(self, currency: str) -> str:
        """Return the unit of the exposed prices, e.g. ``DKK/kWh``."""
        return f"{currency}/{self.price_type}"

    def convert(self, price: float, *, all_in: bool = False) -> float:
        """Return the exposed value of one price given in currency/kWh.

        Args:
            price: A raw spot price excl. VAT, or an all-in consumer price.
            all_in: True if ``price`` already includes VAT and surcharges
                (Stromligning): it is only converted to the unit.

        Returns:
            The price in the configured unit, rounded to the precision.
        """
        per_unit = (
            price * PRICE_IN["kWh"] / PRICE_IN.get(self.price_type, PRICE_IN["kWh"])
        )
        if not all_in:
            per_unit = apply_price_components(per_unit, self.surcharge, self.vat)
        return float(round(per_unit, self.precision))

    def day_series(self, prices: Sequence[float | None]) -> list[float | None]:
        """Return a day's raw prices per exposed interval (per hour if hourly)."""
        return hourly_averages(prices) if self.hourly_average else list(prices)

    def day_prices(
        self, prices: Sequence[float | None], *, all_in: bool = False
    ) -> list[float | None]:
        """Return a day's prices as exposed: per interval, converted.

        Args:
            prices: One price per 15-min slot from local midnight (None if missing).
            all_in: Whether the prices already include VAT (see ``convert``).
        """
        return [
            None if price is None else self.convert(price, all_in=all_in)
            for price in self.day_series(prices)
        ]

    def price_at(
        self,
        prices: Sequence[float | None],
        moment: datetime,
        *,
        all_in: bool = False,
    ) -> float | None:
        """Return the exposed price of the interval containing ``moment``.

        Args:
            prices: The prices of ``moment``'s local day (see ``day_prices``).
            moment: Timezone-aware time within that day.
            all_in: Whether the prices already include VAT (see ``convert``).

        Returns:
            The converted price of the slot, or of the hour's mean with
            ``hourly_average``; None if it is not known.
        """
        series = self.day_series(prices)
        index = slot_index_in_day(moment, interval_minutes=self.interval_minutes)
        price = series[index] if index < len(series) else None
        return None if price is None else self.convert(price, all_in=all_in)

    def forecast_series(
        self, predictions: Iterable[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Return the raw forecast per exposed interval (see ``hourly_forecast``)."""
        if self.hourly_average:
            return hourly_forecast(predictions)
        return slot_forecast(predictions)

    def forecast(self, predictions: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return the forecast as exposed: per interval, prices converted.

        Args:
            predictions: Model predictions (raw spot prices, see ``slot_forecast``).

        Returns:
            ``{"start", "end", "price", "confidence"}`` entries.
        """
        return [
            {**entry, "price": self.convert(entry["price"])}
            for entry in self.forecast_series(predictions)
        ]

    def evaluation(self, rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return evaluated slots (#36) with both prices converted.

        Args:
            rows: ``{"start", "end", "predicted", "actual", "lead_hours"}``
                per slot, prices raw spot (the predictor's ``evaluation``).

        Returns:
            The same entries, ``predicted`` and ``actual`` as exposed.
        """
        return [
            {
                **row,
                "predicted": self.convert(row["predicted"]),
                "actual": self.convert(row["actual"]),
            }
            for row in rows
        ]
