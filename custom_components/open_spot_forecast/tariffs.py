"""The non-spot part of the consumer price, per 15-minute slot (#107).

Stromligning's consumer price excl. VAT is the spot price plus the
supplier's surcharge, the electricity tax, Energinet's net and system
tariffs and the grid company's time-of-use tariff. Its sensors publish the
consumer and the spot price per slot for today and tomorrow, so their
difference is every non-spot component of a slot, excl. VAT. That is the
tariff added to the forecast; its separate tariff sensors only report the
current value.

The model never sees a tariff (#16): ``PriceOutput`` adds it to the
predicted spot price when a price is exposed. A slot past the published
prices takes the latest known day's tariff at the same local time of day:
Danish tariffs follow a fixed daily schedule.
"""

from __future__ import annotations

from bisect import bisect_right
from datetime import UTC, datetime, time

from homeassistant.util import dt as dt_util

from .spot_prices import slot_prices


class TariffSchedule:
    """Tariff (consumer − spot price, excl. VAT) by slot and by local time of day."""

    def __init__(self, by_slot: dict[datetime, float]) -> None:
        """Initialize from the known tariffs, keyed by UTC slot start."""
        self._by_slot = by_slot
        self._by_time: dict[time, float] = {}
        # In time order, so each local time of day keeps its latest tariff
        for start in sorted(by_slot):
            self._by_time[dt_util.as_local(start).time()] = by_slot[start]
        self._times = sorted(self._by_time)

    @classmethod
    def from_prices(
        cls, consumer_data: dict | None, spot_data: dict | None
    ) -> TariffSchedule:
        """Return the schedule of the slots where both prices are known.

        Args:
            consumer_data: Consumer prices excl. VAT
                (``read_stromligning_sensor()``-shaped: ``today``,
                ``tomorrow``, ``day``).
            spot_data: Spot prices excl. VAT (``read_spot_prices()``).

        Returns:
            The schedule; empty (falsy) without a slot known in both.
        """
        spot = {
            start.astimezone(UTC): price for start, _, price in slot_prices(spot_data)
        }
        return cls(
            {
                utc: consumer - spot[utc]
                for start, _, consumer in slot_prices(consumer_data)
                if (utc := start.astimezone(UTC)) in spot
            }
        )

    def __bool__(self) -> bool:
        """Return whether any tariff is known."""
        return bool(self._by_slot)

    def __len__(self) -> int:
        """Return the number of slots with a known tariff."""
        return len(self._by_slot)

    def at(self, start: datetime) -> float:
        """Return the tariff of the slot starting at ``start``.

        Args:
            start: Timezone-aware slot start.

        Returns:
            The slot's own tariff if it is known; else the latest known
            day's tariff at the same local time of day (or the latest
            earlier time of day, for an hour a DST change skipped); 0.0 if
            no tariff is known.
        """
        utc = start.astimezone(UTC)
        if utc in self._by_slot:
            return self._by_slot[utc]
        if not self._times:
            return 0.0
        local = dt_util.as_local(utc).time()
        tariff = self._by_time.get(local)
        if tariff is not None:
            return tariff
        index = bisect_right(self._times, local) - 1
        return self._by_time[self._times[index]]
